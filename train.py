import random
from argparse import ArgumentParser
from os.path import join
import albumentations as A
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from util import build_strategy
from dataset.SphDataset import SubsetSampler
from dataset.SphDataset import Dataset2d, Dataset3d

def parse_arg():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                        default="data/preprocessed")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ndf", type=int, default=16)
    parser.add_argument("--seed", type=int, default=9527)
    parser.add_argument("--initial-labeled", type=int, default=1000)
    parser.add_argument("--budget", type=int, default=2600)
    parser.add_argument("--query", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-size", type=int, default=416)
    parser.add_argument("--forget-weight", type=bool, default=False)
    parser.add_argument("--query-strategy", type=str, default="LeastConfidence")
    parser.add_argument("--query-strategy-param", type=dict,
                        default={"round": 10, "pool_size": 8, "constrative_sampler_size": 20,"difficulty_strategy":"max_entropy"})
    parser.add_argument("--trainer-param", type=dict, default={"num_augmentations": 3})
    args = parser.parse_args()
    args.output_dir = args.query_strategy
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.checkpoint = os.path.join(args.output_dir, "checkpoint")
    if not os.path.exists(args.checkpoint):
        os.mkdir(args.checkpoint)
    return args


def random_seed(seed):
    random.seed(seed)
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


def get_samplers(data_num, initial_labeled):
    data_indice = list(range(data_num))
    np.random.shuffle(data_indice)
    return SubsetSampler(data_indice[:initial_labeled]), SubsetSampler(data_indice[initial_labeled:])


def save_query_plot(folder, labeled_percent, dice_list):
    with open(f"{folder}/result.txt", "w") as fp:
        fp.write("x:")
        fp.write(str(labeled_percent))
        fp.write("\ny:")
        fp.write(str(dice_list))
    plt.plot(labeled_percent, dice_list)
    plt.savefig(f"{folder}/result.jpg")


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
    labeled_sampler, unlabeled_sampler = get_samplers(len(dataset_train), args.initial_labeled)
    dulabeled = torch.utils.data.DataLoader(dataset_train,
                                            batch_size=args.batch_size,
                                            sampler=unlabeled_sampler,
                                            persistent_workers=True,
                                            pin_memory=True,
                                            prefetch_factor=4,
                                            num_workers=args.num_workers)

    dlabeled = torch.utils.data.DataLoader(dataset_train,
                                           batch_size=args.batch_size,
                                           sampler=labeled_sampler,
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

    logger.info(
        f'Initial configuration: len(du): {len(dataloader["unlabeled"].sampler.indices)} '
        f'len(dl): {len(dataloader["labeled"].sampler.indices)} ')

    strategy_type, trainer_type = build_strategy(args.query_strategy)
    trainer = trainer_type(args, logger, writer, args.trainer_param)

    query_strategy = strategy_type(dataloader["unlabeled"], dataloader["labeled"], trainer=trainer,
                                   **args.query_strategy_param)

    loss, iou, dice = trainer.train(dataloader["labeled"], args.epoch, -1)
    logger.info(f"initial model TRAIN | avg_loss: {loss} Dice:{dice} Mean IoU: {iou} ")

    # validation
    dice, meaniou, assd = trainer.valid(dataloader["val"], -1, args.batch_size, args.input_size)
    logger.info(f"initial model EVAl | Dice:{dice} Mean IoU: {meaniou} assd: {assd} ")

    num_dataset = len(dataloader["labeled"].dataset)
    labeled_percent, dice_list = [], []

    ratio = round(len(dataloader["labeled"].sampler.indices) / num_dataset, 4)
    labeled_percent.append(ratio)
    dice_list.append(dice)
    trainer.save(f"{args.checkpoint}/cycle={-1}&labeled={ratio}&dice={dice:.3f}&time={time.time()}.pth")

    cycle = 0
    budget = args.budget
    query = args.query
    total_cycle = (budget // query) + 1
    while budget > 0:
        logger.info(f"cycle {cycle} | budget : {budget} query : {query}")

        if query > budget:
            query = budget
        budget -= query
        cycle += 1

        query_strategy.sample(query)

        logger.info(f'add {query} samplers to labeled dataset')

        # retrain model on updated dataloader
        loss, iou, dice = trainer.train(dataloader["labeled"], args.epoch, cycle)
        logger.info(f"CYCLE {cycle} TRAIN | avg_loss: {loss} avg_dice:{dice} avg_mean_iou: {iou} ")

        dice, meaniou, assd = trainer.valid(dataloader["val"], cycle, args.batch_size, args.input_size)
        logger.info(
            f'Cycle {cycle} EVAl | len(dl): {len(dataloader["labeled"].sampler.indices)} len(du): {len(dataloader["unlabeled"].sampler.indices)} |  avg_dice:{dice} avg_mean_iou: {meaniou} avg_assd: {assd} ')

        ratio = np.round(len(dataloader["labeled"].sampler.indices) / num_dataset, 4)
        labeled_percent.append(ratio)
        dice_list.append(np.round(dice, 4))

        # save checkpoint
        trainer.save(f"{args.checkpoint}/cycle={cycle}&labeled={ratio}&dice={dice:.3f}&time={time.time()}.pth")

        if len(dataloader["unlabeled"].sampler.indices) == 0:
            break

        if args.forget_weight:
            logger.info("forget weight")
            # reset model
            trainer.forget_weight(cycle, total_cycle)
    save_query_plot(args.output_dir, labeled_percent, dice_list)
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
