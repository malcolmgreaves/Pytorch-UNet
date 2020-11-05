import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset, ReconstructDataset
from torch.utils.data import DataLoader, random_split

dir_img = "data/imgs/"
dir_mask = "data/masks/"
dir_checkpoint = "checkpoints/"


def train_net(
    net: nn.Module,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 1,
    lr: float = 0.001,
    save_cp: bool = True,
    img_scale: float = 0.5,
):

    dataset = ReconstructDataset(dir_img, img_scale)
    n_train = len(dataset)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )

    writer = SummaryWriter(comment=f"LR_{lr}_BS_{batch_size}_SCALE_{img_scale}")
    global_step = 0

    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    """
    )

    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.AdamW(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(
            total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img"
        ) as pbar:
            for batch in train_loader:
                imgs = batch["image"]
                assert imgs.shape[1] == net.n_channels, (
                    f"Network has been defined with {net.n_channels} input channels, "
                    f"but loaded images have {imgs.shape[1]} channels. "
                    "Please check that the images are loaded correctly."
                )

                imgs = imgs.to(device=device, dtype=torch.float32)

                imgs_pred = net(imgs)
                import ipdb

                ipdb.set_trace()
                loss = criterion(imgs_pred, imgs)
                epoch_loss += loss.item()
                writer.add_scalar("Loss/train", loss.item(), global_step)

                pbar.set_postfix(**{"loss (batch)": loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                scheduler.step(loss)

                pbar.update(imgs.shape[0])
                global_step += 1

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info("Created checkpoint directory")
            except OSError:
                pass
            fname = f"{dir_checkpoint}CP_epoch{epoch + 1}.pth"
            torch.save(net.state_dict(), fname)
            logging.info(f"Checkpoint {epoch + 1} saved as {fname} !")

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        metavar="E",
        type=int,
        default=5,
        help="Number of epochs",
        dest="epochs",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="B",
        type=int,
        nargs="?",
        default=1,
        help="Batch size",
        dest="batchsize",
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        metavar="LR",
        type=float,
        nargs="?",
        default=0.0001,
        help="Learning rate",
        dest="lr",
    )
    parser.add_argument(
        "-f",
        "--load",
        dest="load",
        type=str,
        default=False,
        help="Load model from a .pth file",
    )
    parser.add_argument(
        "-s",
        "--scale",
        dest="scale",
        type=float,
        default=0.5,
        help="Downscaling factor of the images",
    )
    parser.add_argument(
        "-v",
        "--validation",
        dest="val",
        type=float,
        default=10.0,
        help="Percent of the data that is used as validation (0-100)",
    )

    return parser.parse_args()


def entrypoint():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=1, bilinear=True)
    logging.info(
        f"Network:\n"
        f"\t{net.n_channels} input channels\n"
        f"\t{net.n_classes} output channels (classes)\n"
        f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling'
    )

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f"Model loaded from {args.load}")

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(
            net=net,
            epochs=args.epochs,
            batch_size=args.batchsize,
            lr=args.lr,
            device=device,
            img_scale=args.scale,
        )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), "INTERRUPTED.pth")
        logging.info("Saved interrupt")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == "__main__":
    entrypoint()
