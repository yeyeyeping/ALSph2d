import torch
from monai.metrics import Cumulative, DiceMetric, MeanIoU, SurfaceDistanceMetric
from tqdm import tqdm
import time
from monai.networks.utils import one_hot
from torchvision.utils import make_grid
import numpy as np
from util import label_smooth
from scipy.ndimage import zoom
from util.taalhelper import *
import util.jitfunc as f

SPACING32: float = np.spacing(1, dtype=np.float32)

class NoInfSurfaceDistanceMetric(SurfaceDistanceMetric):

    def get_buffer(self):
        buffer = super().get_buffer()
        return buffer[torch.isinf(buffer) == 0].unsqueeze(1)


class BaseTrainer(object):
    def __init__(self, model, scheduler, logger, writer, criterion, param: dict = None) -> None:
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.optimizer = scheduler.optimizer
        self.device = next(iter(model.parameters())).device
        self.init_metric()
        self.logger = logger
        self.writer = writer
        self.criterion = criterion
        self.additional_param = param

    def writer_scalar(self, niter, cycle, train_loss, mIoU, dice, stage, assd: int = None, prefix: str = None):
        prefix = prefix + "/" if prefix is not None else ""
        self.writer.add_scalar(f"{prefix}cycle{cycle}/{stage}/loss", train_loss, niter)
        self.writer.add_scalar(f"{prefix}cycle{cycle}/{stage}/dice", dice, niter)
        self.writer.add_scalar(f"{prefix}cycle{cycle}/{stage}/mIoU", mIoU, niter)
        if assd is not None:
            self.writer.add_scalar(f"{prefix}cycle{cycle}/{stage}/assd", assd, niter)

    def writer_image(self, pred, cycle, niter, img, mask, stage, prefix: str = None):
        pred, img, mask = pred.detach().cpu(), img.detach().cpu(), mask.detach().cpu()
        img = (img - img.min()) / (img.max() - img.min())
        pred, img, mask = pred.repeat(1, 3, 1, 1), img.repeat(1, 3, 1, 1), mask.repeat(1, 3, 1, 1)
        prefix = prefix + "/" if prefix is not None else ""
        image_grid = make_grid(tensor=pred,
                               nrow=pred.shape[0])
        self.writer.add_image(f"{prefix}cycle{cycle}/{stage}/pred", image_grid, niter)
        img_mask = np.concatenate([img, mask])
        raw_imgs = make_grid(tensor=torch.from_numpy(img_mask), nrow=pred.shape[0])
        self.writer.add_image(f"{prefix}cycle{cycle}/{stage}/raw", raw_imgs, niter)

    def init_metric(self):
        self.train_loss, self.batch_time, self.data_time = Cumulative(), Cumulative(), Cumulative()
        self.dice_metric, self.meaniou_metric, self.assd_metric = DiceMetric(include_background=False), MeanIoU(
            include_background=False), NoInfSurfaceDistanceMetric(include_background=False, symmetric=True)

    def batch_forward(self, img, mask):
        output = self.model(img).softmax(dim=1)
        loss = self.criterion(output, mask)
        return output, loss

    def train(self, dataloader, epochs, cycle):
        for epoch in range(epochs):
            self.train_loss.reset(), self.batch_time.reset(), self.data_time.reset(), self.dice_metric.reset(), self.meaniou_metric.reset()
            tbar = tqdm(dataloader)
            tlc = time.time()
            self.model.train()
            for btidx, (img, mask) in enumerate(tbar):
                self.data_time.append(time.time() - tlc)
                img, mask = img.to(self.device), mask.to(self.device)
                mask_onehot = one_hot(mask, 2)

                output, loss = self.batch_forward(img, mask_onehot)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                loss_item = loss.cpu().item()

                self.train_loss.append(loss_item)
                self.batch_time.append(time.time() - tlc)

                tlc = time.time()

                bin_mask = output.argmax(dim=1).unsqueeze(1)

                dice, miou = self.dice_metric(y_pred=bin_mask, y=mask_onehot), self.meaniou_metric(
                    y_pred=bin_mask, y=mask_onehot)
                dice, miou = dice[dice.isnan() == 0].mean(), miou[miou.isnan() == 0].mean()
                tbar.set_description(
                    f"CYCLE {cycle} TRAIN {epoch}| Loss:{loss_item:.3f}  Dice:{dice:.2f} Mean IoU: {miou:.2f} "
                    f"|B {self.batch_time.get_buffer().mean():.2f}) |D {self.data_time.get_buffer().mean():.2f}")

                if btidx % 50 == 0:
                    niter = epoch * len(dataloader)
                    self.logger.info(
                        f"CYCLE {cycle} TRAIN {epoch} iter {niter}| Loss:{loss_item:.3f}  Dice:{dice:.2f} Mean IoU: {miou:.2f}")
                    self.writer_scalar(niter, cycle, self.train_loss.get_buffer().mean(),
                                       self.meaniou_metric.aggregate().item(),
                                       self.dice_metric.aggregate().item(), "Train")
                    self.writer_image(bin_mask, cycle, niter, img, mask, "Train")

        avg_loss, mean_iou, avg_dice = self.train_loss.get_buffer().mean(), self.meaniou_metric.aggregate().item(), self.dice_metric.aggregate().item()
        tbar.set_description(
            f"CYCLE {cycle} TRAIN AVG| Loss:{avg_loss:.3f}  Dice:{avg_dice:.3f} Mean IoU: {mean_iou:.3f}")

        return avg_loss, mean_iou, avg_dice

    @torch.no_grad()
    def valid(self, dataloader, cycle, batch_size, input_size):
        self.model.eval()
        tbar = tqdm(dataloader)
        for idx, (img, mask) in enumerate(tbar):
            self.dice_metric.reset(), self.meaniou_metric.reset(), self.assd_metric.reset()

            pred_volume = np.empty((0, img.shape[-2], img.shape[-1]), dtype=np.float32)
            img, mask = img[0], mask[0]
            h, w = img.shape[-2], img.shape[-1]
            for batch in range(0, img.shape[0], batch_size):
                last = batch + batch_size
                batch_slices = img[batch:] if last >= img.shape[0] else img[batch:last]

                batch_slices = zoom(batch_slices, (1, 1, input_size / h, input_size / w), order=0,
                                    mode='nearest')
                batch_slices = torch.from_numpy(batch_slices).to(self.device)
                batch_pred_mask = self.model(batch_slices).argmax(dim=1).cpu()
                batch_pred_mask = zoom(batch_pred_mask, (1, h / input_size, w / input_size), order=0,
                                       mode='nearest')
                pred_volume = np.concatenate([pred_volume, batch_pred_mask])
                torch.cuda.empty_cache()

            mask_onehot = one_hot(mask, 2)
            volume_pred_mask = torch.from_numpy(label_smooth(pred_volume)).unsqueeze(1)
            dice = self.dice_metric(y_pred=volume_pred_mask, y=mask_onehot)
            iou = self.meaniou_metric(y_pred=volume_pred_mask, y=mask_onehot)
            assd = self.assd_metric(y_pred=volume_pred_mask, y=mask_onehot)
            dice, iou = dice[dice.isnan() == 0].mean(), iou[iou.isnan() == 0].mean()
            assd = assd[(assd.isnan() == 0) & assd.isfinite()].mean()

            tbar.set_description(
                f"CYCLE {cycle} EVAl | Dice:{dice:.3f} Mean IoU: {iou:.2f} asd: {assd:.2f} ")

        avg_dice, avg_iou, avg_assd = \
            np.round(self.dice_metric.aggregate().item(), 3), \
                np.round(self.meaniou_metric.aggregate().item(), 2), \
                np.round(self.assd_metric.aggregate().item(), 2)

        tbar.set_description(
            f"CYCLE {cycle} EVAl AVG| Dice:{avg_dice:.3f} Mean IoU: {avg_iou:.3f} asd: {avg_assd:.3f} ")
        return avg_dice, avg_iou, avg_assd


class TTATrainer(BaseTrainer):

    def batch_forward(self, img, mask):
        output, dice_loss = super().batch_forward(img, mask)
        all_output = augments_forward(img, self.model, output, int(self.additional_param["num_augmentations"]),
                                      self.device)
        consistency_loss = torch.mean(f.JSD(all_output, SPACING32))
        loss = (consistency_loss + dice_loss) / 2
        return output, loss
