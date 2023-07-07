import random
from argparse import ArgumentParser
from os.path import join

import logging
import numpy as np
import os
import time
import torch
from tensorboardX import SummaryWriter
from util import build_strategy
from util import get_dataloader, save_query_plot


def parse_arg():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                        default="data/ACDCprecessed")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ndf", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--initial-labeled", type=float, default=0.1)
    parser.add_argument("--budget", type=int, default=0.2)
    parser.add_argument("--query", type=int, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--forget-weight", type=bool, default=False)
    parser.add_argument("--query-strategy", type=str, default="LeastConfidence")
    parser.add_argument("--query-strategy-param", type=str,
                        default='{"round": 10, "distance_measure": "cosine", "pool_size": 8,\
                                 "constrative_sampler_size": 20, "difficulty_strategy": "max_entropy"}')
    parser.add_argument("--trainer-param", type=str, default='{"num_augmentations": 3}')
    args = parser.parse_args()
    if args.output_dir == "":
        args.output_dir = args.query_strategy
    args.query_strategy_param = eval(args.query_strategy_param)
    args.trainer_param = eval(args.trainer_param)
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

    loss, dice, clsdice = trainer.train(dataloader, args.epoch, -1)

    logger.info(f"initial model TRAIN | avg_loss: {loss} Dice:{dice}{clsdice}")

    # validation
    dice, meaniou, assd = trainer.valid(dataloader["test"], -1, args.batch_size)
    logger.info(f"initial model EVAl | Dice:{dice} Mean IoU: {meaniou} assd: {assd} ")

    num_dataset = len(dataloader["labeled"].dataset)
    labeled_percent, dice_list = [], []

    ratio = round(len(dataloader["labeled"].sampler.indices) / num_dataset, 4)
    labeled_percent.append(ratio)
    dice_list.append(dice)
    trainer.save(f"{args.checkpoint}/cycle={-1}&labeled={ratio}&dice={dice:.3f}&time={time.time()}.pth")

    cycle = 0
    budget = int(args.budget * num_dataset)
    query = int(args.query * num_dataset)
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
        loss, dice, clsdice = trainer.train(dataloader, args.epoch, cycle)
        if loss == dice == None:
            break
        logger.info(f"CYCLE {cycle} TRAIN | avg_loss: {loss} avg_dice:{dice}{clsdice} ")

        dice, meaniou, assd = trainer.valid(dataloader["test"], cycle, args.batch_size)
        logger.info(
            f'Cycle {cycle} EVAl | len(dl): {len(dataloader["labeled"].sampler.indices)} len(du): {len(dataloader["unlabeled"].sampler.indices)} |  avg_dice:{dice} avg_mean_iou: {meaniou} avg_assd: {assd} ')

        ratio = np.round(len(dataloader["labeled"].sampler.indices) / num_dataset, 4)
        labeled_percent.append(ratio)
        dice_list.append(np.round(dice, 4))
        save_query_plot(args.output_dir, labeled_percent, dice_list)
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
    logger.warning(args)
    al_cycle(args=args, logger=logger)
