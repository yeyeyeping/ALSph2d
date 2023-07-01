import torch
from monai.metrics import Cumulative, DiceMetric, MeanIoU, SurfaceDistanceMetric
from torch import nn
from tqdm import tqdm
import time
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
from util import get_current_consistency_weight, linear_rampup


# need a function to unit the way of model‘s output
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
        self.train_loss, self.batch_time = Cumulative(), Cumulative()
        self.dice_metric, self.meaniou_metric = DiceMetric(include_background=False, ignore_empty=False), MeanIoU(
            include_background=False, ignore_empty=False)

    def batch_forward(self, img, onehot_mask):
        output, _ = self.model(img)
        output = output.softmax(1)
        loss = self.criterion(output, onehot_mask)
        return output, loss

    def train(self, dataloader, epochs, cycle):
        self.train_loss.reset(), self.batch_time.reset(), self.dice_metric.reset(), self.meaniou_metric.reset()
        dataloader = dataloader["labeled"]
        tbar = tqdm(dataloader)
        for epoch in range(epochs):
            tlc = time.time()
            self.model.train()
            for btidx, (img, mask) in enumerate(tbar):
                img, mask = img.to(self.args.device), mask.to(self.args.device)
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

                dice, miou = self.dice_metric(y_pred=bin_mask, y=mask_onehot).mean(), self.meaniou_metric(
                    y_pred=bin_mask, y=mask_onehot).mean()
                tbar.set_description(
                    f"CYCLE {cycle} TRAIN {epoch}| Loss:{loss_item:.3f}  Dice:{dice:.2f} Mean IoU: {miou:.2f} "
                    f"|B {self.batch_time.get_buffer().mean():.2f})")

                if btidx % 50 == 0:
                    niter = epoch * len(dataloader)
                    self.logger.info(
                        f"CYCLE {cycle} TRAIN {epoch} iter {niter}| Loss:{loss_item:.3f}  Dice:{dice:.2f} Mean IoU: {miou:.2f}")
                    self.writer_scalar(niter, cycle, self.train_loss.gezt_buffer().mean(),
                                       self.meaniou_metric.aggregate().item(),
                                       self.dice_metric.aggregate().item(), "Train")
                    self.writer_image(bin_mask, cycle, niter, img, mask, "Train")

        avg_loss, mean_iou, avg_dice = self.train_loss.get_buffer().mean(), self.meaniou_metric.aggregate().item(), self.dice_metric.aggregate().item()
        tbar.set_description(
            f"CYCLE {cycle} TRAIN AVG| Loss:{avg_loss:.3f}  Dice:{avg_dice:.3f} Mean IoU: {mean_iou:.3f}")

        return avg_loss, mean_iou, avg_dice

    def save(self, ckpath):
        torch.save(self.model.state_dict(), ckpath)

    @torch.no_grad()
    def valid(self, dataloader, cycle, batch_size, input_size):
        self.model.eval()
        tbar = tqdm(dataloader)
        dice_his, iou_his, assd_his = [], [], []
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

            dice_his.append(dice)
            iou_his.append(iou)
            assd_his.append(assd)

        avg_dice, avg_iou, avg_assd = \
            np.round(np.mean(np.array(dice_his)), 3), \
                np.round(np.mean(np.array(iou_his)), 3), \
                np.round(np.mean(np.array(assd_his)), 3)

        tbar.set_description(
            f"CYCLE {cycle} EVAl AVG| Dice:{avg_dice:.3f} Mean IoU: {avg_iou:.3f} asd: {avg_assd:.3f} ")
        return avg_dice, avg_iou, avg_assd


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


ContrastiveTrainer = CoresetTrainer

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


class URPCTrainer(BaseTrainer):

    def build_model(self):
        from model.network_urpc import UNet_URPC
        model = UNet_URPC(1, 2).to(self.args.device)
        model.apply(lambda param: initialize_weights(param, 1))
        return model

    def train(self, dataloader, epochs, cycle):
        from dataset.dataset import TwoStreamBatchSampler
        from torch.utils.data import DataLoader
        labeled_idx, unlabeled_idx = dataloader["labeled"].sampler.indices, dataloader["unlabeled"].sampler.indices
        btsize = dataloader["labeled"].batch_size // 2
        batch_sampler = TwoStreamBatchSampler(labeled_idx, unlabeled_idx, btsize, btsize)
        trainloader = DataLoader(dataloader["labeled"].dataset, batch_sampler=batch_sampler,
                                 num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4, )

        tbar = tqdm(trainloader)
        kl_distance = nn.KLDivLoss(reduction='none')
        for epoch in range(epochs):

            self.model.train()
            for btidx, (img, mask) in enumerate(tbar):
                tlc = time.time()
                img, mask = img.to(self.args.device), mask.to(self.args.device)
                mask_onehot = one_hot(mask[:btsize], 2)

                outputs, outputs_aux1, outputs_aux2, outputs_aux3 = self.model(
                    img)
                outputs_soft = torch.softmax(outputs, dim=1)
                outputs_aux1_soft = torch.softmax(outputs_aux1, dim=1)
                outputs_aux2_soft = torch.softmax(outputs_aux2, dim=1)
                outputs_aux3_soft = torch.softmax(outputs_aux3, dim=1)

                loss_sup = (self.criterion(outputs_soft[:btsize], mask_onehot) \
                            + self.criterion(outputs_aux1_soft[:btsize], mask_onehot) \
                            + self.criterion(outputs_aux2_soft[:btsize], mask_onehot) \
                            + self.criterion(outputs_aux3_soft[:btsize], mask_onehot)) / 4

                preds = (outputs_soft + outputs_aux1_soft +
                         outputs_aux2_soft + outputs_aux3_soft) / 4

                variance_main = torch.sum(kl_distance(
                    torch.log(outputs_soft[btsize:]), preds[btsize:]), dim=1, keepdim=True)
                exp_variance_main = torch.exp(-variance_main)

                variance_aux1 = torch.sum(kl_distance(
                    torch.log(outputs_aux1_soft[btsize:]), preds[btsize:]), dim=1, keepdim=True)
                exp_variance_aux1 = torch.exp(-variance_aux1)

                variance_aux2 = torch.sum(kl_distance(
                    torch.log(outputs_aux2_soft[btsize:]), preds[btsize:]), dim=1, keepdim=True)
                exp_variance_aux2 = torch.exp(-variance_aux2)

                variance_aux3 = torch.sum(kl_distance(
                    torch.log(outputs_aux3_soft[btsize:]), preds[btsize:]), dim=1, keepdim=True)
                exp_variance_aux3 = torch.exp(-variance_aux3)

                consistency_weight = get_current_consistency_weight(epoch // epochs)
                consistency_dist_main = (
                                                preds[btsize:] - outputs_soft[btsize:]) ** 2

                consistency_loss_main = torch.mean(
                    consistency_dist_main * exp_variance_main) / (torch.mean(exp_variance_main) + 1e-8) + torch.mean(
                    variance_main)

                consistency_dist_aux1 = (preds[btsize:] - outputs_aux1_soft[btsize:]) ** 2
                consistency_loss_aux1 = torch.mean(
                    consistency_dist_aux1 * exp_variance_aux1) / (torch.mean(exp_variance_aux1) + 1e-8) + torch.mean(
                    variance_aux1)

                consistency_dist_aux2 = (
                                                preds[btsize:] - outputs_aux2_soft[btsize:]) ** 2
                consistency_loss_aux2 = torch.mean(
                    consistency_dist_aux2 * exp_variance_aux2) / (torch.mean(exp_variance_aux2) + 1e-8) + torch.mean(
                    variance_aux2)

                consistency_dist_aux3 = (
                                                preds[btsize:] - outputs_aux3_soft[btsize:]) ** 2
                consistency_loss_aux3 = torch.mean(
                    consistency_dist_aux3 * exp_variance_aux3) / (torch.mean(exp_variance_aux3) + 1e-8) + torch.mean(
                    variance_aux3)

                consistency_loss = (consistency_loss_main + consistency_loss_aux1 +
                                    consistency_loss_aux2 + consistency_loss_aux3) / 4

                loss = loss_sup + consistency_weight * consistency_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                loss_item = loss.cpu().item()

                self.train_loss.append(loss_item)
                self.batch_time.append(time.time() - tlc)

                tlc = time.time()

                bin_mask = preds[:btsize].argmax(dim=1).unsqueeze(1)

                dice, miou = self.dice_metric(y_pred=bin_mask, y=mask_onehot).mean(), self.meaniou_metric(
                    y_pred=bin_mask, y=mask_onehot).mean()
                tbar.set_description(
                    f"CYCLE {cycle} TRAIN {epoch}| Loss:{loss_item:.3f}  Dice:{dice:.2f} Mean IoU: {miou:.2f} "
                    f"|B {self.batch_time.get_buffer().mean():.2f})")

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


class MGTrainer(BaseTrainer):

    def __init__(self, args, logger, writer, param: dict = None) -> None:
        self.param = {
            "class_num": 2,
            "in_chns": 1,
            "block_type": "UNetBlock",
            "feature_chns": [64, 128, 256, 512],
            "feature_grps": [4, 4, 4, 4, 1],
            "norm_type": "group_norm",
            "acti_func": "relu",
            "dropout": True,
            "depth_sep_deconv": False,
            "deep_supervision": False,
        }
        super().__init__(args, logger, writer, param)

    def build_model(self):
        from model.MGNet.MGNet import MGNet
        model = MGNet(self.param).to(self.args.device)
        model.apply(lambda param: initialize_weights(param, 1))
        return model

    def semi_train(self, labeled_loader, unlabeled_loader, epochs, cycle):
        from dataset.dataset import TwoStreamBatchSampler
        from torch.utils.data import DataLoader
        labeled_idx, unlabeled_idx = labeled_loader.sampler.indices, unlabeled_loader.sampler.indices
        btsize = labeled_loader.batch_size // 2
        if len(unlabeled_idx) < btsize:
            return None, None, None
        batch_sampler = TwoStreamBatchSampler(labeled_idx, unlabeled_idx, labeled_loader.batch_size, btsize)
        trainloader = DataLoader(labeled_loader.dataset, batch_sampler=batch_sampler,
                                 num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4, )
        self.train_loss.reset(), self.batch_time.reset(), self.dice_metric.reset(), self.meaniou_metric.reset()
        for epoch in range(epochs):
            tbar = tqdm(trainloader)
            self.model.train()
            for btidx, (img, mask) in enumerate(tbar):
                tlc = time.time()
                img, mask = img.to(self.args.device), mask.to(self.args.device)
                mask_onehot = one_hot(mask[:btsize], 2)
                outputlist = self.model(img)
                # Group x Batch x C x H x W
                output = torch.stack(outputlist).softmax(dim=2)

                # consistency loss for all data
                consistency_loss = torch.mean(f.JSD(output))

                labeled_output, unlabeled_output = output[:, :btsize], output[:, btsize:],

                # dicece loss for labeled data
                labeled_mask = mask_onehot[None].repeat(len(labeled_output), 1, 1, 1, 1)
                G, N, C, H, W = labeled_output.shape
                outshape = [G * N, C, H, W]
                labeled_output = torch.reshape(labeled_output, shape=outshape)
                labeled_mask = torch.reshape(labeled_mask, shape=outshape)
                dice_loss = self.criterion(labeled_output, labeled_mask)

                loss = (dice_loss + consistency_loss) / 2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                loss_item = loss.cpu().item()

                self.train_loss.append(loss_item)

                preds = torch.mean(output, dim=0)
                bin_mask = preds[:btsize].argmax(dim=1).unsqueeze(1)

                dice, miou = self.dice_metric(y_pred=bin_mask, y=mask_onehot).mean(), self.meaniou_metric(
                    y_pred=bin_mask, y=mask_onehot).mean()

                b = time.time() - tlc
                self.batch_time.append(b)

                tbar.set_description(
                    f"CYCLE {cycle} TRAIN {epoch}| Loss:{loss_item:.3f}  Dice:{dice:.2f} Mean IoU: {miou:.2f} "
                    f"|B {b:.2f}) ")

                if btidx % 50 == 0:
                    niter = epoch * len(trainloader)
                    self.logger.info(
                        f"CYCLE {cycle} TRAIN {epoch} iter {niter}| Loss:{loss_item:.3f}  Dice:{dice:.2f} Mean IoU: {miou:.2f}")
                    self.writer_scalar(niter, cycle, self.train_loss.get_buffer().mean(),
                                       self.meaniou_metric.aggregate().item(),
                                       self.dice_metric.aggregate().item(), "Train")
                    self.writer_image(bin_mask, cycle, niter, img, mask, "Train")

        avg_loss, mean_iou, avg_dice = self.train_loss.get_buffer().mean(), self.meaniou_metric.aggregate().item(), self.dice_metric.aggregate().item()
        tbar.set_description(
            f"CYCLE {cycle} TRAIN AVG| Loss:{avg_loss:.3f}  Dice:{avg_dice:.3f} Mean IoU: {mean_iou:.3f} |B {self.batch_time.get_buffer().mean():.2f}")

        return avg_loss, mean_iou, avg_dice

    def pseudo_train(self, pseudo_loader, epochs, cycle):
        from monai.losses import DiceLoss
        from torch.nn import CrossEntropyLoss
        self.train_loss.reset(), self.batch_time.reset(), self.dice_metric.reset(), self.meaniou_metric.reset()
        dice_loss = DiceLoss(include_background=True, softmax=False, reduction="none")
        ce_loss = CrossEntropyLoss(reduction="none")
        for epoch in range(epochs):
            self.model.train()
            tbar = tqdm(pseudo_loader)
            for btidx, (img, mask) in enumerate(tbar):
                tlc = time.time()
                img, mask = img.to(self.args.device), mask.to(self.args.device)
                squeeze_mask = mask.squeeze(1).long()
                pred = torch.stack(self.model(img))
                onehot_mask = one_hot(mask, 2)
                group_loss = []
                for group_pred in pred:
                    group_ce = ce_loss(group_pred, squeeze_mask).mean(dim=[-1, -2])
                    group_dice = dice_loss(group_pred.softmax(1), onehot_mask).mean(dim=[-1, -2, -3])
                    group_loss.append((group_ce + group_dice) / 2)
                loss_tesnor = torch.stack(group_loss)
                _, idx = torch.sort(loss_tesnor, dim=1)
                # 0.25 percent of the samplers are selected to Coteaching
                idx_select = idx[:, :loss_tesnor.shape[1] // 4][torch.randperm(loss_tesnor.shape[0])]
                loss = torch.gather(loss_tesnor, 1, idx_select).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                loss_item = loss.cpu().item()

                self.train_loss.append(loss_item)

                preds = torch.mean(pred, dim=0)
                bin_mask = preds.argmax(dim=1).unsqueeze(1)

                dice, miou = self.dice_metric(y_pred=bin_mask, y=onehot_mask).mean(), self.meaniou_metric(
                    y_pred=bin_mask, y=onehot_mask).mean()

                b = time.time() - tlc
                self.batch_time.append(b)

                tbar.set_description(
                    f"CYCLE {cycle} TRAIN(pseudo) {epoch}| Loss:{loss_item:.3f}  Dice:{dice:.2f} Mean IoU: {miou:.2f} "
                    f"|B {b:.2f}) ")

                if btidx % 50 == 0:
                    niter = epoch * len(pseudo_loader)
                    self.logger.info(
                        f"CYCLE {cycle} TRAIN(pseudo) {epoch} iter {niter}| Loss:{loss_item:.3f}  Dice:{dice:.2f} Mean IoU: {miou:.2f}")
                    self.writer_scalar(niter, cycle, self.train_loss.get_buffer().mean(),
                                       self.meaniou_metric.aggregate().item(),
                                       self.dice_metric.aggregate().item(), "Train")
                    self.writer_image(bin_mask, cycle, niter, img, mask, "Train")

        avg_loss, mean_iou, avg_dice = self.train_loss.get_buffer().mean(), self.meaniou_metric.aggregate().item(), self.dice_metric.aggregate().item()
        tbar.set_description(
            f"CYCLE {cycle} TRAIN(pseudo ) AVG| Loss:{avg_loss:.3f}  Dice:{avg_dice:.3f} Mean IoU: {mean_iou:.3f} |B {self.batch_time.get_buffer().mean():.2f}")
        return avg_loss, mean_iou, avg_dice

    @torch.no_grad()
    def valid(self, dataloader, cycle, batch_size, input_size=416):
        self.model.eval()
        tbar = tqdm(dataloader)
        dice_his, iou_his, assd_his = [], [], []
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
                output = torch.stack(self.model(batch_slices)).mean(0)
                batch_pred_mask = output.argmax(dim=1).cpu()
                batch_pred_mask = zoom(batch_pred_mask, (1, h / input_size, w / input_size), order=0,
                                       mode='nearest')
                batch_pred.append(batch_pred_mask)

            pred_volume = np.concatenate(batch_pred)
            del batch_pred
            dice, iou, assd = get_metric(pred_volume, np.asarray(mask.squeeze(1)))

            tbar.set_description(
                f"CYCLE {cycle} EVAl | Dice:{dice:.2f} Mean IoU: {iou:.2f} asd: {assd:.2f} ")

            dice_his.append(dice)
            iou_his.append(iou)
            assd_his.append(assd)

        avg_dice, avg_iou, avg_assd = \
            np.round(np.mean(np.array(dice_his)), 3), \
                np.round(np.mean(np.array(iou_his)), 3), \
                np.round(np.mean(np.array(assd_his)), 3)

        tbar.set_description(
            f"CYCLE {cycle} EVAl AVG| Dice:{avg_dice:.3f} Mean IoU: {avg_iou:.3f} asd: {avg_assd:.3f} ")
        return avg_dice, avg_iou, avg_assd


class OnlineMGTrainer(BaseTrainer):

    def __init__(self, args, logger, writer, param: dict = None) -> None:
        self.param = {
            "class_num": 2,
            "in_chns": 1,
            "block_type": "UNetBlock",
            "feature_chns": [64, 128, 256, 512],
            "feature_grps": [4, 4, 4, 4, 1],
            "norm_type": "group_norm",
            "acti_func": "relu",
            "dropout": True,
            "depth_sep_deconv": False,
            "deep_supervision": False,
        }
        super().__init__(args, logger, writer, param)

    def build_model(self):
        from model.MGNet.MGNet import MGNet
        model = MGNet(self.param).to(self.args.device)
        model.apply(lambda param: initialize_weights(param, 1))
        return model

    def semi_train(self, labeled_loader, unlabeled_loader, epochs, cycle):
        from dataset.dataset import TwoStreamBatchSampler
        from torch.utils.data import DataLoader
        labeled_idx, unlabeled_idx = labeled_loader.sampler.indices, unlabeled_loader.sampler.indices
        btsize = labeled_loader.batch_size // 2
        if len(unlabeled_idx) < btsize:
            return None, None, None
        batch_sampler = TwoStreamBatchSampler(labeled_idx, unlabeled_idx, labeled_loader.batch_size, btsize)
        trainloader = DataLoader(labeled_loader.dataset, batch_sampler=batch_sampler,
                                 num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4, )
        self.train_loss.reset(), self.batch_time.reset(), self.dice_metric.reset(), self.meaniou_metric.reset()
        for epoch in range(epochs):
            tbar = tqdm(trainloader)
            self.model.train()
            for btidx, (img, mask) in enumerate(tbar):
                tlc = time.time()
                img, mask = img.to(self.args.device), mask.to(self.args.device)
                mask_onehot = one_hot(mask[:btsize], 2)
                outputlist = self.model(img)
                # Group x Batch x C x H x W
                output = torch.stack(outputlist).softmax(dim=2)

                # consistency loss for all data
                consistency_loss = torch.mean(f.JSD(output))

                labeled_output, unlabeled_output = output[:, :btsize], output[:, btsize:],

                # dicece loss for labeled data
                labeled_mask = mask_onehot[None].repeat(len(labeled_output), 1, 1, 1, 1)
                G, N, C, H, W = labeled_output.shape
                outshape = [G * N, C, H, W]
                labeled_output = torch.reshape(labeled_output, shape=outshape)
                labeled_mask = torch.reshape(labeled_mask, shape=outshape)
                dice_loss = self.criterion(labeled_output, labeled_mask)
                # pseudo learning for unlabeled data
                batched_output = torch.reshape(unlabeled_output, shape=outshape)
                pseudo_label = one_hot(batched_output.detach().argmax(dim=1).unsqueeze(1), 2)
                pseudo_label = pseudo_label.reshape([G, N, C, H, W])
                permed_label = pseudo_label[torch.randperm(len(pseudo_label))]
                pseudo_loss = self.criterion(unlabeled_output[0], permed_label[0])
                for idx in range(1, len(permed_label)):
                    pseudo_loss += self.criterion(unlabeled_output[idx], permed_label[idx])

                pseudo_loss = pseudo_loss / len(unlabeled_output)
                alpha = linear_rampup(epoch, epochs)
                # loss = dice_loss + consistency_loss + alpha*pseudo_loss
                loss = dice_loss + alpha * pseudo_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                loss_item = loss.cpu().item()

                self.train_loss.append(loss_item)

                preds = torch.mean(output, dim=0)
                bin_mask = preds[:btsize].argmax(dim=1).unsqueeze(1)

                dice, miou = self.dice_metric(y_pred=bin_mask, y=mask_onehot).mean(), self.meaniou_metric(
                    y_pred=bin_mask, y=mask_onehot).mean()

                b = time.time() - tlc
                self.batch_time.append(b)

                tbar.set_description(
                    f"CYCLE {cycle} TRAIN {epoch}| Loss: {loss_item:.3f} D:{dice_loss.item():.3f} P: {pseudo_loss.item():.3f}| Dice: {dice:.2f} Mean IoU: {miou:.2f} "
                    f"|B {b:.2f}) ")

                if btidx % 50 == 0:
                    niter = epoch * len(trainloader)
                    self.logger.info(
                        f"CYCLE {cycle} TRAIN {epoch} iter {niter}| Loss:{loss_item:.3f}  Dice:{dice:.2f} Mean IoU: {miou:.2f}")
                    self.writer_scalar(niter, cycle, self.train_loss.get_buffer().mean(),
                                       self.meaniou_metric.aggregate().item(),
                                       self.dice_metric.aggregate().item(), "Train")
                    self.writer_image(bin_mask, cycle, niter, img, mask, "Train")

        avg_loss, mean_iou, avg_dice = self.train_loss.get_buffer().mean(), self.meaniou_metric.aggregate().item(), self.dice_metric.aggregate().item()
        tbar.set_description(
            f"CYCLE {cycle} TRAIN AVG| Loss:{avg_loss:.3f}  Dice:{avg_dice:.3f} Mean IoU: {mean_iou:.3f} |B {self.batch_time.get_buffer().mean():.2f}")

        return avg_loss, mean_iou, avg_dice

    def train(self, dataloader, epochs, cycle):
        labeled_loader, unlabeled_loader, pseudo_loader = dataloader.values()
        return self.semi_train(labeled_loader, unlabeled_loader, epochs, cycle)

    @torch.no_grad()
    def valid(self, dataloader, cycle, batch_size, input_size=416):
        self.model.eval()
        tbar = tqdm(dataloader)
        dice_his, iou_his, assd_his = [], [], []
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
                output = torch.stack(self.model(batch_slices)).mean(0)
                batch_pred_mask = output.argmax(dim=1).cpu()
                batch_pred_mask = zoom(batch_pred_mask, (1, h / input_size, w / input_size), order=0,
                                       mode='nearest')
                batch_pred.append(batch_pred_mask)

            pred_volume = np.concatenate(batch_pred)
            del batch_pred
            dice, iou, assd = get_metric(pred_volume, np.asarray(mask.squeeze(1)))

            tbar.set_description(
                f"CYCLE {cycle} EVAl | Dice:{dice:.2f} Mean IoU: {iou:.2f} asd: {assd:.2f} ")

            dice_his.append(dice)
            iou_his.append(iou)
            assd_his.append(assd)

        avg_dice, avg_iou, avg_assd = \
            np.round(np.mean(np.array(dice_his)), 3), \
                np.round(np.mean(np.array(iou_his)), 3), \
                np.round(np.mean(np.array(assd_his)), 3)

        tbar.set_description(
            f"CYCLE {cycle} EVAl AVG| Dice:{avg_dice:.3f} Mean IoU: {avg_iou:.3f} asd: {avg_assd:.3f} ")
        return avg_dice, avg_iou, avg_assd
