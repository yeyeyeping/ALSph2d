import numpy as np

from util import get_dataloader, save_query_plot
from util import parse_config, random_seed, init_logger
import importlib


def getclass(module, classname):
    module = importlib.import_module(module)
    assert hasattr(module, classname), f"{classname} Not Implement"
    return module.__dict__[classname]


def get_trainer(config, dataloader, logger):
    query_strategy = config["AL"]["query_strategy"]
    strategy_section = config["all_strategy"][query_strategy]

    trainer_class = getclass(strategy_section["trainer"]["module"],
                             strategy_section["trainer"]["class"])
    trainer_obj = trainer_class(config, logger=logger, *strategy_section["trainer"]["additional_param"])

    query_strategy = query_strategy if "class" not in strategy_section else query_strategy["class"]

    strategy_class = getclass(strategy_section["module"], query_strategy)
    strategy_obj = strategy_class(dataloader, trainer=trainer_obj, *strategy_section["additional_param"])
    logger.info(f"strategy:{type(strategy_obj)}  param:{strategy_section['additional_param']}")

    logger.info(f"trainer:{type(trainer_obj)} param: {strategy_section['trainer']['additional_param']}")

    return strategy_obj, trainer_obj


def main(config):
    logger = init_logger(config)
    dataloader = get_dataloader(config)
    num_dataset = len(dataloader["labeled"].dataset)
    labeled_percent, dice_list = [], []
    logger.info(
        f'Initial configuration: len(du): {len(dataloader["unlabeled"].sampler.indices)} '
        f'len(dl): {len(dataloader["labeled"].sampler.indices)} ')

    query_strategy, trainer = get_trainer(config, dataloader, logger)

    # initialize model
    val_metric = trainer.train(dataloader, 0)

    initial_ratio = np.round(len(dataloader["labeled"].sampler.indices) / num_dataset, 2)
    labeled_percent.append(initial_ratio)
    dice_list.append(np.round(val_metric['avg_fg_dice'], 4))

    valid_dice = "[" + ' '.join("{0:.4f}".format(x) for x in val_metric['class_dice']) + "]"
    logger.info(
        f"initial model TRAIN | avg_loss: {val_metric['loss']} Dice:{val_metric['avg_fg_dice']} {valid_dice}")

    budget = int(config["AL"]["budget"] * num_dataset)
    query = int(config["AL"]["query"] * num_dataset)

    cycle = 0
    while budget > 0:
        logger.info(f"cycle {cycle} | budget : {budget} query : {query}")

        if query > budget:
            query = budget
        budget -= query
        cycle += 1

        query_strategy.sample(query)
        logger.info(f'add {query} samplers to labeled dataset')

        # retrain model on updated dataloader
        val_metric = trainer.train(dataloader, cycle)
        valid_dice = "[" + ' '.join("{0:.4f}".format(x) for x in val_metric['class_dice']) + "]"
        logger.info(
            f"Cycle{cycle} TRAIN | avg_loss: {val_metric['loss']} Dice:{val_metric['avg_fg_dice']} {valid_dice}")

        ratio = np.round(len(dataloader["labeled"].sampler.indices) / num_dataset, 2)
        labeled_percent.append(ratio)
        dice_list.append(np.round(val_metric['avg_fg_dice'], 4))
        save_query_plot(config["Training"]["output_dir"], labeled_percent, dice_list)

        # save checkpoint
        if len(dataloader["unlabeled"].sampler.indices) == 0:
            break

        trainer.finish()


if __name__ == "__main__":
    config = parse_config()

    random_seed(config)

    main(config)
