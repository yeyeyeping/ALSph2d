from os.path import join

import numpy as np
import time
from tensorboardX import SummaryWriter
from util import build_strategy
from util import get_dataloader, save_query_plot
from util import parse_arg, random_seed, init_logger

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

    logger.info(args)

    al_cycle(args=args, logger=logger)
