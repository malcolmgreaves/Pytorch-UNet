import argparse
import logging
import os
import sys
import gc
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from unet.unet_model import UNetReconstruct
from utils.dataset import ReconstructDataset

dir_checkpoint = "checkpoints/"


def refresh_cuda_memory(gpu_device: torch.device) -> None:
    """
    Re-allocate all cuda memory to help alleviate fragmentation
    From: https://github.com/pytorch/pytorch/issues/31252
    """
    # Run a full garbage collect first so any dangling tensors are released
    gc.collect()

    cpu_device = torch.device("cpu")

    objs_w_tensors = []
    # Then move all tensors to the CPU
    for obj in tqdm(gc.get_objects(), desc="Re-organizing tensors in GPU memory"):
        if not isinstance(obj, torch.Tensor):
            continue
        obj.data = obj.data.to(cpu_device)
        if isinstance(obj, torch.nn.Parameter) and obj.grad is not None:
            obj.grad = obj.grad.to(cpu_device)
        objs_w_tensors.append(obj)

    # Now empty the cache to flush the allocator
    torch.cuda.empty_cache()

    # Finally move the tensors back to their associated GPUs
    for obj in objs_w_tensors:
        obj.data = obj.data.to(gpu_device)
        if isinstance(obj, torch.nn.Parameter) and obj.grad is not None:
            obj.grad = obj.grad.to(gpu_device)


def train_net(
    embeding_dir: Path,
    net: nn.Module,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 1,
    lr: float = 0.001,
    save_cp: bool = True,
    img_scale: float = 0.5,
):

    dataset = ReconstructDataset(
        embeding_dir=embeding_dir,
        scale=img_scale,
        device=device,
        is_grayscale=True,
    )
    n_train = len(dataset)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False
    )

    writer = SummaryWriter(comment=f"LR_{lr}_BS_{batch_size}_SCALE_{img_scale}")
    global_step = 0

    logging.info(
        f"""Starting training:
        Embeding dir w/ images: {embeding_dir}
        Epochs:                 {epochs}
        Batch size:             {batch_size}
        Learning rate:          {lr}
        Training size:          {n_train}
        Checkpoints:            {save_cp}
        Device:                 {device.type}
        Images scaling:         {img_scale}
    """
    )

    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.AdamW(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2)
    criterion = nn.MSELoss()

    # https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/
    scaler = torch.cuda.amp.GradScaler()
    ooms = 0
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0.0
        with tqdm(
            total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img"
        ) as pbar:
            count = 0
            for i, batch in enumerate(train_loader):
                imgs = batch["image"]
                assert imgs.shape[1] == net.n_channels, (
                    f"Network has been defined with {net.n_channels} input channels, "
                    f"but loaded images have {imgs.shape[1]} channels. "
                    "Please check that the images are loaded correctly."
                )
                count = min(n_train, count + batch_size)

                try:
                    optimizer.zero_grad()

                    imgs = imgs.to(device=device, dtype=torch.float32)
                    imgs_pred = net(imgs)
                    # with torch.cuda.amp.autocast(): # 1.6.1+ only :*(
                    loss = criterion(imgs_pred, imgs)
                    epoch_loss += loss.item()
                    writer.add_scalar("Loss/train", loss.item(), global_step)

                    pbar.set_postfix(
                        **{
                            "loss (batch)": loss.item(),
                            "loss (epoch)": epoch_loss / count,
                        }
                    )

                    scaler.scale(loss).backward()  # loss.backward()
                    nn.utils.clip_grad_value_(net.parameters(), 0.1)
                    scaler.step(optimizer)  # optimizer.step()

                    scaler.update()

                    pbar.update(imgs.shape[0])
                    global_step += 1

                except RuntimeError as e:
                    if "cuda out of memory" in str(e).lower():
                        print(
                            f"WARNING: attempting to recover from OOM [#{ooms+1}] in forward/backward pass",
                            file=sys.stderr,
                        )
                        ooms += 1
                        optimizer.zero_grad()
                        del batch
                        refresh_cuda_memory(device)
                    else:
                        raise e
        avg_epoch_loss = epoch_loss / n_train
        logging.info(
            f"Epoch {epoch + 1} has average loss of {avg_epoch_loss} (sum: {epoch_loss})"
        )
        scheduler.step(epoch_loss)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info("Created checkpoint directory")
            except OSError:
                pass
            fname = f"{dir_checkpoint}CP_epoch{epoch + 1}.pth"
            torch.save(net.state_dict(), fname)
            logging.info(f"Checkpoint {epoch + 1} saved as {fname} !")
        if ooms > 0:
            print(f"{ooms} CUDA out-of-memory errors encountered")
    writer.close()


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=lambda s: Path(s).absolute(),
        help="Input dir of images (nested structure)",
        required=True,
    )
    parser.add_argument(
        "--skip-top-residual",
        action="store_true",
        help="If present, skips the top-level skip connection. Otherwise is a vanilla U-Net",
        required=False,
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

    if not args.input_dir.is_dir():
        raise ValueError(
            "Need to provide input directory via --input-dir (not file or nothing!)"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNetReconstruct(
        n_channels=1,
        bilinear=True,
        skip_top_residual=args.skip_top_residual,
    )
    logging.info(
        f"Network:\n"
        f"\t{net.n_channels} channels\n"
        f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling'
    )

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f"Model loaded from {args.load}")

    net.to(device=device)
    summary(net, input_size=_input_size(args, device))
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(
            embeding_dir=args.input_dir,
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


def _input_size(args, device):
    _input_size = next(
        iter(
            DataLoader(
                dataset=ReconstructDataset(
                    embeding_dir=args.input_dir,
                    scale=args.scale,
                    device=device,
                    is_grayscale=True,
                ),
                batch_size=args.batchsize,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
        )
    )["image"].shape[1:]
    return _input_size


if __name__ == "__main__":
    entrypoint()
