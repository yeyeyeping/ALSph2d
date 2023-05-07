from monai.metrics import Cumulative, DiceMetric, MeanIoU, SurfaceDistanceMetric
from tqdm import tqdm
import time
from util import AverageMeter
from monai.networks.utils import one_hot
from torchvision.utils import make_grid
import numpy as np
from model.LossPredictionModule import LossPredLoss
from scipy.ndimage import zoom
from util.taalhelper import *
import util.jitfunc as f
from monai.losses import DiceCELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from model import build_model, initialize_weights, UNetWithDropout
from util.metric import get_metric
from model import UNetWithFeature
from pymic.util.evaluation_seg import binary_dice, binary_iou


class BaseTrainer(object):
    def __init__(self, args, logger, writer, param: dict = None) -> None:
        super().__init__()
        self.args = args
        self.logger = logger
        self.writer = writer

        self.additional_param = param
        self.model = self.build_model()

        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_sched(self.optimizer)

        self.criterion = self.build_criterion()

        self.init_metric()

    def build_criterion(self):
        return DiceCELoss(include_background=True, softmax=False)

    def build_optimizer(self):
        return Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def build_sched(self, optimizer):
        return MultiStepLR(optimizer, milestones=[0.4 * self.args.epoch, 0.7 * self.args.epoch], gamma=0.5)

    def build_model(self):
        model = UNetWithFeature(1, 2, self.args.ndf).to(self.args.device)
        model.apply(lambda param: initialize_weights(param, 1))
        return model

    def forget_weight(self, cycle, total_cycle):
        # self.model.apply(lambda param: initialize_weights(param, p=1 - cycle / total_cycle))
        self.model.apply(lambda param: initialize_weights(param, p=1))

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
        self.train_loss, self.batch_time, self.data_time = AverageMeter(), AverageMeter(), AverageMeter()
        self.dice_metric, self.meaniou_metric, self.assd_metric = AverageMeter(), AverageMeter(), AverageMeter()

    def batch_forward(self, img, onehot_mask):
        output, _ = self.model(img)
        output = output.softmax(1)
        loss = self.criterion(output, onehot_mask)
        return output, loss

    def train(self, dataloader, epochs, cycle):
        tbar = tqdm(dataloader)
        self.train_loss.reset(), self.batch_time.reset(), self.data_time.reset(), self.dice_metric.reset(), self.meaniou_metric.reset()
        for epoch in range(epochs):
            tlc = time.time()
            self.model.train()
            for btidx, (img, mask) in enumerate(tbar):
                self.data_time.update(time.time() - tlc)
                img, mask = img.to(self.args.device), mask.to(self.args.device)
                mask_onehot = one_hot(mask, 2)

                output, loss = self.batch_forward(img, mask_onehot)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                loss_item = loss.cpu().item()

                self.train_loss.update(loss_item, output.size(0))
                self.batch_time.update(time.time() - tlc)

                tlc = time.time()

                bin_mask = output.argmax(dim=1).detach().cpu()
                cpu_mask = mask.squeeze(1).detach().cpu()
                dice, miou = binary_dice(bin_mask, cpu_mask), binary_iou(bin_mask, cpu_mask)
                self.dice_metric.update(dice, output.size(0))
                self.meaniou_metric.update(miou, output.size(0))

                tbar.set_description(
                    f"CYCLE {cycle} TRAIN {epoch}| Loss:{loss_item:.3f}  Dice:{dice:.2f} Mean IoU: {miou:.2f} "
                    f"|B {self.batch_time.average}) |D {self.data_time.average}")

                if btidx % 50 == 0:
                    niter = epoch * len(dataloader)
                    self.logger.info(
                        f"CYCLE {cycle} TRAIN {epoch} iter {niter}| Loss:{loss_item:.3f}  Dice:{dice:.2f} Mean IoU: {miou:.2f}")
                    self.writer_scalar(niter, cycle, self.train_loss.average,
                                       self.meaniou_metric.average,
                                       self.dice_metric.average, "Train")
                    self.writer_image(bin_mask.unsqueeze(1), cycle, niter, img, mask, "Train")

        tbar.set_description(
            f"CYCLE {cycle} TRAIN AVG| Loss:{self.train_loss.avg}  Dice:{self.dice_metric.avg} Mean IoU: {self.meaniou_metric.avg}")

        return self.train_loss.average, self.dice_metric.average, self.meaniou_metric.avg

    def save(self, ckpath):
        torch.save(self.model.state_dict(), ckpath)

    @torch.no_grad()
    def valid(self, dataloader, cycle, batch_size, input_size):
        self.model.eval()
        tbar = tqdm(dataloader)
        self.meaniou_metric.reset()
        self.dice_metric.reset()
        self.assd_metric.reset()
        for idx, (img, mask) in enumerate(tbar):
            img, mask = img[0], mask[0]
            h, w = img.shape[-2], img.shape[-1]
            batch_pred = []
            for batch in range(0, img.shape[0], batch_size):
                last = batch + batch_size
                batch_slices = img[batch:] if last >= img.shape[0] else img[batch:last]

                batch_slices = zoom(batch_slices, (1, 1, input_size / h, input_size / w), order=0,
                                    mode='nearest')
                batch_slices = torch.from_numpy(batch_slices).to(self.args.device)
                output, _ = self.model(batch_slices)
                batch_pred_mask = output.argmax(dim=1).cpu()
                batch_pred_mask = zoom(batch_pred_mask, (1, h / input_size, w / input_size), order=0,
                                       mode='nearest')
                batch_pred.append(batch_pred_mask)

            pred_volume = np.concatenate(batch_pred)
            del batch_pred
            dice, iou, assd = get_metric(pred_volume, np.asarray(mask.squeeze(1)))

            tbar.set_description(
                f"CYCLE {cycle} EVAl | Dice:{dice:.3f} Mean IoU: {iou:.2f} asd: {assd:.2f} ")

            self.dice_metric.update(dice)
            self.meaniou_metric.update(iou)
            self.assd_metric.update(assd)

        tbar.set_description(
            f"CYCLE {cycle} EVAl AVG| Dice:{self.dice_metric.avg} Mean IoU: {self.meaniou_metric.avg} assd: {self.assd_metric.avg} ")
        return self.dice_metric.avg, self.meaniou_metric.avg, self.assd_metric.avg


class TTATrainer(BaseTrainer):

    def batch_forward(self, img, onehot_mask):
        output, dice_loss = super().batch_forward(img, onehot_mask)
        all_output = augments_forward(img, self.model, output, int(self.additional_param["num_augmentations"]),
                                      self.args.device)
        consistency_loss = torch.mean(f.JSD(all_output))
        loss = (consistency_loss + dice_loss) / 2
        return output, loss


class BALDTrainer(BaseTrainer):
    def build_model(self):
        model = UNetWithDropout(1, 2, self.args.ndf).to(self.args.device)
        model.apply(lambda x: initialize_weights(x, 1))
        return model


class LearningLossTrainer(BaseTrainer):
    def build_model(self):
        from model.LossPredictionModule import LossPredModule
        self.loss_predition_module = LossPredModule().to(self.args.device)
        self.loss_predition_module.apply(lambda x: initialize_weights(x, 1))
        model = UNetWithFeature(1, 2, self.args.ndf).to(self.args.device)
        model.apply(lambda x: initialize_weights(x, 1))
        return model

    def build_optimizer(self):
        import itertools
        return Adam(
            params=itertools.chain(self.model.parameters(), self.loss_predition_module.parameters()),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay)

    def build_criterion(self):
        from monai.losses import DiceLoss
        return DiceLoss(include_background=True, softmax=False, reduction="none")

    def batch_forward(self, img, onehot_mask):
        self.loss_predition_module.train()
        output, features = self.model(img)
        output = output.softmax(dim=1)

        dice_loss = torch.mean(self.criterion(output, onehot_mask), dim=1).view((img.shape[0],))
        pred_loss = self.loss_predition_module(features).view(img.shape[0], )
        loss_pred_loss = LossPredLoss(pred_loss, dice_loss)
        return output, (torch.mean(dice_loss) + loss_pred_loss) / 2


class CoresetTrainer(BaseTrainer):
    def build_model(self):
        model = UNetWithFeature(1, 2, self.args.ndf).to(self.args.device)
        model.apply(lambda x: initialize_weights(x, 1))
        return model

    def batch_forward(self, img, onehot_mask):
        output, _ = self.model(img)
        output = output.softmax(1)
        loss = self.criterion(output, onehot_mask)
        return output, loss


ConstrativeTrainer = CoresetTrainer

from torch.nn import functional as F


class DEALTrainer(BaseTrainer):

    def mask_mean(self, x, mask):
        return x[mask].mean() if mask.sum() > 0 else torch.tensor([0.]).cuda()

    def cal_class_weights(self, right_mask, error_mask):
        pixel_num = torch.tensor([right_mask.sum(), error_mask.sum()]).float().cuda()
        class_weights = 1 / torch.sqrt(pixel_num + 1)
        class_weights = class_weights / class_weights.sum()
        return class_weights

    def weight_ce(self, soft_mask, target_error_mask):
        loss = F.binary_cross_entropy(soft_mask.squeeze(1), target_error_mask, reduction='none')

        error_mask = target_error_mask.bool()
        right_mask = ~error_mask
        weights = self.cal_class_weights(right_mask, error_mask)

        loss = self.mask_mean(loss, right_mask) * weights[0] + self.mask_mean(loss, error_mask) * weights[1]
        return loss

    def generate_target_error_mask(self, output, target):
        pred = torch.argmax(output, dim=1)
        target_error_mask = (pred != target).float()  # error=1
        return target_error_mask

    def build_model(self):
        from model.pam import PAM
        self.pam = PAM().to(self.args.device)
        self.pam.apply(lambda x: initialize_weights(x, 1))

        model = UNetWithFeature(1, 2, self.args.ndf).to(self.args.device)
        model.apply(lambda x: initialize_weights(x, 1))
        return model

    def build_criterion(self):
        self.aux_loss = self.weight_ce
        return DiceCELoss(include_background=True, softmax=False)

    def build_optimizer(self):
        import itertools
        return Adam(
            params=itertools.chain(self.model.parameters(), self.pam.parameters()),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay)

    def batch_forward(self, img, onehot_mask):
        output, _ = self.model(img)
        output = output.softmax(1)
        dice_loss = self.criterion(output.softmax(1), onehot_mask)

        difficult_map, _ = self.pam(output)
        target_error_mask = self.generate_target_error_mask(output, onehot_mask[:, 1])
        error_pred_loss = self.aux_loss(difficult_map, target_error_mask)
        return output, 0.5 * (dice_loss + error_pred_loss)
