"""Loguru logger"""

import os
import sys
import time
from loguru import logger
from pathlib import Path


def setup_logger(name,
                 save_dir,
                 prefix="",
                 use_std=True,
                 out_file=None):
    """Setup loguru logger"""

    logger.remove()
    fmt = f"<green>{{time:YYYY-MM-DD HH:mm:ss.SSS}}</green> | " \
          f"<cyan>{name}.{{extra[ext]}}</cyan> | " \
          f"<lvl>{{level}}</lvl> | " \
          f"<lvl>{{message}}</lvl>"

    # logger to file
    if save_dir:
        timestamp = time.strftime(".%m_%d_%H_%M_%S")
        log_file = Path(save_dir)/f'log{prefix}{timestamp}.txt'
        if out_file:
            log_file = out_file
        print(f'Save log to file {log_file}')
        logger.add(log_file, level="INFO", format=fmt)

    # logger to std stream
    if use_std:
        logger.add(sys.stdout,
                   level="INFO",
                   format=fmt)


def get_logger(ext=""):
    logger1 = logger.bind(ext=ext)
    return logger1


if __name__ == "__main__":
    output_dir = '/Users/sfhan/Documents/project/voxelPoint/tmp'
    log_file = os.path.join(output_dir, "log1.txt")
    name = 'BoxNet'

    setup_logger(name=name, save_dir=output_dir, prefix="test")
    train_logger = get_logger(ext="train")
    train_logger.info('Hello info')
    val_logger = get_logger(ext="val")
    val_logger.info('Hello info')
    train_logger.info(f'Test.into')
    default_logger = get_logger()
    default_logger.info('Hello info')

