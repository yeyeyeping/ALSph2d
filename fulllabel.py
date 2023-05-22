from argparse import ArgumentParser
from os.path import join
import albumentations as A
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, MeanIoU, Cumulative, SurfaceDistanceMetric
from monai.networks.utils import one_hot
from scipy.ndimage import zoom
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from util import build_strategy
from util import label_smooth
from dataset.SphDataset import SubsetSampler
from dataset.SphDataset import Dataset2d, Dataset3d
from model import build_model, initialize_weights
from util.trainer import BaseTrainer


def parse_arg():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                        default="data/preprocessed")
    parser.add_argument("--output-dir", type=str, default="FullLabeled")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ndf", type=int, default=16)
    parser.add_argument("--seed", type=int, default=9527)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-size", type=int, default=416)
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.checkpoint = os.path.join(args.output_dir, "checkpoint")
    if not os.path.exists(args.checkpoint):
        os.mkdir(args.checkpoint)
    return args


def random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def init_logger(args):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    fh = logging.FileHandler(f"{args.output_dir}/{time.time()}.log")
    fh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(fh)
    logger.info(str(args))
    return logger


def get_dataloader(args):
    train_transform = A.Compose([
        A.Resize(512, 512),
        A.CropNonEmptyMaskIfExists(args.input_size, args.input_size, p=1),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(p=0.2),
    ])
    dataset_train, dataset_val = Dataset2d(datafolder=os.path.join(args.data_dir, "train"),
                                           transform=train_transform), Dataset3d(
        folder=os.path.join(args.data_dir, "val"))

    dulabeled = torch.utils.data.DataLoader(dataset_train,
                                            batch_size=args.batch_size,
                                            persistent_workers=True,
                                            pin_memory=True,
                                            prefetch_factor=4,
                                            num_workers=args.num_workers)

    dlabeled = torch.utils.data.DataLoader(dataset_train,
                                           batch_size=args.batch_size,
                                           persistent_workers=True,
                                           pin_memory=True,
                                           prefetch_factor=4,
                                           num_workers=args.num_workers)

    dval = torch.utils.data.DataLoader(dataset_val,
                                       batch_size=1,
                                       persistent_workers=True,
                                       pin_memory=True,
                                       prefetch_factor=4,
                                       num_workers=args.num_workers)
    return {
        "labeled": dlabeled,
        "unlabeled": dulabeled,
        "val": dval
    }


def al_cycle(args, logger):
    writer = SummaryWriter(join(args.output_dir, "tensorboard"))
    dataloader = get_dataloader(args)

    trainer = BaseTrainer(args, logger, writer, {})
    for i in range(40):
        loss, iou, dice = trainer.train(dataloader["labeled"], args.epoch, -1)

        logger.info(f"model TRAIN | avg_loss: {loss} Dice:{dice} Mean IoU: {iou} ")
        # validation
        dice, meaniou, assd = trainer.valid(dataloader["val"], -1, args.batch_size, args.input_size)
        logger.info(f"model EVAl | Dice:{dice} Mean IoU: {meaniou} assd: {assd} ")
        torch.save(trainer.model.state_dict(),
                   f"{args.checkpoint}/epoch={i * args.epoch}labeled1&dice={dice:.3f}&time={time.time()}.pth")
    writer.flush()
    writer.close()


if __name__ == "__main__":
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = False
    cudnn.deterministic = True

    args = parse_arg()
    random_seed(args.seed)
    logger = init_logger(args)

    al_cycle(args=args, logger=logger)
