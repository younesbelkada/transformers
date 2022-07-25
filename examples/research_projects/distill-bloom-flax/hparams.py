import os
import random
from dataclasses import dataclass

import simple_parsing


@dataclass
class Hparams:
    """
    Hyperparameters for the experiments. This includes
        - training hyper parameters
        - wandb parameters
    """

    # Global params
    seed: int = 42

    # wandb params
    wandb_entity: str = "distill-bloom"
    wandb_project: str = "test-small-distillation"
    wandb_logs: str = "/home/sanchitgandhi/cache/younes_files/wandb_logs"
    root_dir: str = os.getcwd()

    # teacher / student params
    teacher_path: str = "bigscience/bloom-1b3"
    student_path: str = "bigscience/distill-bloom-1b3"

    # Dataset params
    epochs: int = 2
    batch_size: int = 8
    num_workers: int = 0
    path_bin_data: str = "/home/sanchitgandhi/cache/younes_files/binarized_data"
    max_seq_len: int = 512
    vocab_size: int = 250880

    # Learning params
    learning_rate: float = 0.0001

    # optimizer params
    optimizer_name: str = "adam"
    step_size: int = 1


@dataclass
class Parameters:
    """Global parameters options."""

    hparams: Hparams = Hparams()

    @classmethod
    def parse(cls):
        parser = simple_parsing.ArgumentParser()
        parser.add_arguments(cls, dest="parameters")
        args = parser.parse_args()
        instance: Parameters = args.parameters
        return instance
