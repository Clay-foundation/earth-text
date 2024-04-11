import os
import random

import dotenv
import hydra
import lightning.pytorch as pl
import numpy as np
import omegaconf
from loguru import logger
from omegaconf import DictConfig



@hydra.main(
    version_base="1.1",
    config_path="configs",
    config_name="train.yaml",
)

def main(config: DictConfig):
    print (config)

    dataloader = hydra.utils.instantiate(config.dataloader)

    print ("XX", dataloader)


if __name__ == "__main__":
    main()