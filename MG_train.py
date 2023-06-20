import random
from argparse import ArgumentParser
from os.path import join

from util.trainer import OnlineMGTrainer
from util.query_strategy import OnlineMGQuery
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from tensorboardX import SummaryWriter
from util import save_query_plot

from util import get_dataloader


def parse_arg():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                        default="data/preprocessed")
    parser.add_argument("--output-dir", type=str, default="")
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
    parser.add_argument("--query-strategy", type=str, default="OnlineMG")
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
    from util.trainer import MGTrainer
    from util.query_strategy import MGQuery
    writer = SummaryWriter(join(args.output_dir, "tensorboard"))
    dataloader = get_dataloader(args, with_pseudo=True)

    logger.info(
        f'Initial configuration: len(du): {len(dataloader["unlabeled"].sampler.indices)} '
        f'len(dl): {len(dataloader["labeled"].sampler.indices)} ')

    trainer = MGTrainer(args, logger, writer, args.trainer_param)

    query_strategy = MGQuery(dataloader, trainer, args.query)

    loss, iou, dice = trainer.semi_train(dataloader["labeled"], dataloader["unlabeled"], args.epoch, -1)
    logger.info(f"initial model TRAIN | avg_loss: {loss} Dice:{dice} Mean IoU: {iou} ")
    #
    # # validation
    dice, meaniou, assd = trainer.valid(dataloader["test"], -1, args.batch_size, args.input_size)
    logger.info(f"initial model EVAl | Dice:{dice} Mean IoU: {meaniou} assd: {assd} ")
    #
    num_dataset = len(dataloader["labeled"].dataset)
    labeled_percent, dice_list = [], []
    #
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
        s_loss, s_iou, s_dice = trainer.semi_train(dataloader["labeled"], dataloader["unlabeled"], args.epoch, cycle)
        logger.info(f"CYCLE {cycle} TRAIN | avg_loss: {s_loss} avg_dice:{s_dice} avg_mean_iou: {s_iou} ")
        dice, meaniou, assd = trainer.valid(dataloader["test"], cycle, args.batch_size, args.input_size)
        logger.info(
            f'Cycle {cycle} EVAl | len(dl): {len(dataloader["labeled"].sampler.indices)} len(du): {len(dataloader["unlabeled"].sampler.indices)} |  avg_dice:{dice} avg_mean_iou: {meaniou} avg_assd: {assd} ')

        loss, iou, dice = trainer.pseudo_train(dataloader["pseudo"], args.epoch, cycle)
        logger.info(f"CYCLE {cycle} TRAIN(pseudo) | avg_loss: {loss} avg_dice:{dice} avg_mean_iou: {iou} ")
        dice, meaniou, assd = trainer.valid(dataloader["test"], cycle, args.batch_size, args.input_size)
        logger.info(
            f'Cycle {cycle} EVAl(pseudo) | len(dl): {len(dataloader["labeled"].sampler.indices)} len(du): {len(dataloader["unlabeled"].sampler.indices)} |  avg_dice:{dice} avg_mean_iou: {meaniou} avg_assd: {assd} ')

        ratio = np.round(len(dataloader["labeled"].sampler.indices) / num_dataset, 4)
        labeled_percent.append(ratio)
        dice_list.append(np.round(dice, 4))
        save_query_plot(args.output_dir, labeled_percent, dice_list)
        # save checkpoint
        trainer.save(f"{args.checkpoint}/cycle={cycle}&labeled={ratio}&dice={dice:.3f}&time={time.time()}.pth")

        if s_loss == s_iou == s_dice == None:
            break

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
