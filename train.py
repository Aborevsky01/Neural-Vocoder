import argparse
import warnings

import numpy as np
from torch.utils.data import DataLoader

from configs import *
from datasets import *
from gan import HifiGAN
from trainer import *

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(train_config, args):
    buffer = get_data_to_buffer(train_config)
    train_dataset = BufferDataset(buffer)
    val_dataset = EvalDataset(train_config)

    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_expand_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = HifiGAN(train_config)
    model = model.to(train_config.device)
    trainer = Trainer(
        model,
        train_config,
        train_config.device,
        {'train': train_loader, 'val': val_loader}
    )
    if args.resume is not None:
        trainer.resume_checkpoint(args.resume)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    parser.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    args = parser.parse_args()
    train_config = TrainConfig()
    if args.device is not None:
        train_config.device = args.device
    main(train_config, args)
