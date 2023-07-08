import copy

from monai.losses import DiceCELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime
import time
from torch.optim import Adam
from model import UNetWithFeature, initialize_weights
from torch.utils.data import DataLoader
from monai.networks.utils import one_hot
import torch
import numpy as np
from util.metric import get_classwise_dice, get_multi_class_metric
from scipy.ndimage import zoom
from tensorboardX import SummaryWriter
from os.path import join


class BaseTrainer:
    def __init__(self, config, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.device = self.config["Training"]["device"]
        self.logger = kwargs["logger"]
        self.additional_param = kwargs
        self.model = self.build_model()
        self.max_val_dice = 0
        self.criterion = self.build_criterion()
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_sched(self.optimizer)
        self.glob_it = 0
        self.best_model_wts = None
        self.summ_writer = SummaryWriter(join(config["Training"]["output_dir"], "tensorboard"))

    def build_model(self):
        model = UNetWithFeature(1, self.config["Network"]["classnum"], self.config["Network"]["ndf"]).to(self.device)
        model.apply(lambda param: initialize_weights(param, 1))
        return model

    def build_optimizer(self):
        return Adam(self.model.parameters(),
                    lr=self.config["Training"]["lr"],
                    weight_decay=self.config["Training"]["weight_decay"])

    def build_criterion(self):
        return DiceCELoss(include_background=True, softmax=False)

    def build_sched(self, optimizer):
        return ReduceLROnPlateau(optimizer, mode="max",
                                 factor=self.config["Training"]["lr_gamma"],
                                 patience=self.config["Training"]["ReduceLROnPlateau_patience"])

    def write_scalars(self, train_scalars, valid_scalars, lr_value, glob_it):
        loss_scalar = {'train': train_scalars['loss'], 'valid': valid_scalars['loss']}
        dice_scalar = {'train': train_scalars['avg_fg_dice'], 'valid': valid_scalars['avg_fg_dice']}
        self.summ_writer.add_scalars('loss', loss_scalar, glob_it)
        self.summ_writer.add_scalars('dice', dice_scalar, glob_it)
        self.summ_writer.add_scalars('lr', {"lr": lr_value}, glob_it)
        class_num = self.config['Network']['classnum']
        for c in range(class_num):
            cls_dice_scalar = {'train': train_scalars['class_dice'][c], \
                               'valid': valid_scalars['class_dice'][c]}
            self.summ_writer.add_scalars(f'class_{c}_dice', cls_dice_scalar, glob_it)

        train_dice = "[" + ' '.join("{0:.4f}".format(x) for x in train_scalars['class_dice']) + "]"
        self.logger.info(
            f"train loss {train_scalars['loss']:.4f}, avg foreground dice {train_scalars['avg_fg_dice']:.4f} {train_dice}")

        valid_dice = "[" + ' '.join("{0:.4f}".format(x) for x in valid_scalars['class_dice']) + "]"
        self.logger.info(
            f"valid loss {valid_scalars['loss']:.4f}, avg foreground dice {valid_scalars['avg_fg_dice']:.4f} {valid_dice}")

    def forward_for_query(self):
        pass

    def batch_forward(self, img, mask, to_onehot_y=False):
        output, _ = self.model(img)
        output = output.softmax(1)
        if to_onehot_y:
            mask = one_hot(mask, self.config["Network"]["classnum"])
        loss = self.criterion(output, mask)
        return output, loss

    def training(self, dataloader: DataLoader):
        iter_valid = self.config["Training"]["iter_valid"]
        classnum = self.config["Network"]["classnum"]
        self.model.train()
        train_loss = 0
        train_dice_list = []
        for it in range(iter_valid):
            try:
                img, mask = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(dataloader)
                img, mask = next(self.train_iter)

            img, mask = img.to(self.device), mask.to(self.device)
            onehot_mask = one_hot(mask, classnum)

            self.optimizer.zero_grad()

            output, loss = self.batch_forward(img, onehot_mask)
            loss.backward()

            self.optimizer.step()
            train_loss += loss.item()
            preds = output.argmax(1).unsqueeze(1)
            bin_mask = one_hot(preds, classnum)
            soft_y = onehot_mask.permute(0, 2, 3, 1).reshape((-1, 4))
            predict = bin_mask.permute(0, 2, 3, 1).reshape((-1, 4))
            dice_tesnsor = get_classwise_dice(predict, soft_y).cpu().numpy()
            train_dice_list.append(dice_tesnsor)
        train_avg_loss = train_loss / iter_valid
        train_cls_dice = np.asarray(train_dice_list).mean(axis=0)
        train_avg_dice = train_cls_dice[1:].mean()
        train_scalers = {'loss': train_avg_loss, 'avg_fg_dice': train_avg_dice, \
                         'class_dice': train_cls_dice}
        return train_scalers

    @torch.no_grad()
    def validation(self, dataloader):
        self.model.eval()
        classnum = self.config["Network"]["classnum"]
        batch_size = self.config["Dataset"]["batch_size"]
        input_size = self.config["Dataset"]["input_size"]

        dice_his, valid_loss = [], []

        for idx, (img, mask) in enumerate(dataloader):
            img, mask = img[0], mask[0]
            h, w = img.shape[-2], img.shape[-1]
            batch_pred = []
            volume_loss = 0
            for batch in range(0, img.shape[0], batch_size):
                last = batch + batch_size
                last = last if last < img.shape[0] else None
                batch_slices, mask_slices = img[batch:last], mask[batch:last]

                batch_slices = zoom(batch_slices, (1, 1, input_size / h, input_size / w), order=2,
                                    mode='nearest')
                mask_slices = zoom(mask_slices, (1, 1, input_size / h, input_size / w), order=0,
                                   mode='nearest')

                batch_slices = torch.from_numpy(batch_slices).to(self.device)
                mask_slices = torch.from_numpy(mask_slices).to(self.device)

                output, loss = self.batch_forward(batch_slices, mask_slices, to_onehot_y=True)
                volume_loss += loss.item()
                batch_pred_mask = output.argmax(dim=1).cpu()
                batch_pred_mask = zoom(batch_pred_mask, (1, h / input_size, w / input_size), order=0,
                                       mode='nearest')
                batch_pred.append(batch_pred_mask)

            pred_volume = np.concatenate(batch_pred)
            del batch_pred
            dice, _, _ = get_multi_class_metric(pred_volume,
                                                np.asarray(mask.squeeze(1)),
                                                classnum, include_backgroud=True, )
            valid_loss.append(volume_loss)
            dice_his.append(dice)
        valid_avg_loss = np.asarray(valid_loss).mean()

        valid_cls_dice = np.asarray(dice_his).mean(axis=0)
        valid_avg_dice = valid_cls_dice[1:].mean()

        valid_scalers = {'loss': valid_avg_loss, 'avg_fg_dice': valid_avg_dice, \
                         'class_dice': valid_cls_dice}
        return valid_scalers

    def train(self, dataloader, cycle):
        train_loader = dataloader["labeled"]
        valid_loder = dataloader["test"]

        iter_max = self.config["Training"]["iter_max"]
        iter_valid = self.config["Training"]["iter_valid"]
        early_stop = self.config["Training"]["early_stop_patience"]

        if cycle > 0:
            bestwts_last = f"{self.config['Training']['checkpoint_dir']}/c{cycle - 1}_best{self.max_val_dice:.4f}.pt"
            ckpoint = torch.load(bestwts_last, map_location=self.device)
            self.model.load_state_dict(ckpoint['model_state_dict'])
            self.optimizer.load_state_dict(ckpoint['optimizer_state_dict'])

        self.max_val_dice = 0
        max_performance_it = 0

        self.train_iter = iter(train_loader)
        start_it = self.glob_it
        for it in range(0, iter_max, iter_valid):
            lr_value = self.optimizer.param_groups[0]['lr']
            t0 = time.time()
            train_scalars = self.training(train_loader)
            t1 = time.time()
            valid_scalars = self.validation(valid_loder)
            t2 = time.time()

            self.scheduler.step(valid_scalars["avg_fg_dice"])

            self.glob_it += iter_valid

            self.logger.info(f"\n{str(datetime.datetime.now())[:-7]} iteration {self.glob_it}")
            self.logger.info(f"learning rate {lr_value}")
            self.logger.info(f"training/validation time:{t1 - t0:.4f}/{t2 - t1:.4f}")

            self.write_scalars(train_scalars, valid_scalars, lr_value, self.glob_it)

            if valid_scalars["avg_fg_dice"] > self.max_val_dice:
                max_performance_it = self.glob_it
                self.max_val_dice = valid_scalars["avg_fg_dice"]
                self.best_model_wts = {
                    'model_state_dict': copy.deepcopy(self.model.state_dict()),
                    'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
                }

            if self.glob_it - max_performance_it > early_stop:
                self.logger.info("The training is early stopped")
                break
        # best
        save_path = f"{self.config['Training']['checkpoint_dir']}/c{cycle}_best{self.max_val_dice:.4f}.pt"
        torch.save(self.best_model_wts, save_path)

        # latest
        save_path = f"{self.config['Training']['checkpoint_dir']}/c{cycle}_g{self.glob_it}_l{self.glob_it - start_it}_latest{valid_scalars['avg_fg_dice']:.4f}.pt"
        save_dict = {'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}
        torch.save(save_dict, save_path)
        return valid_scalars
