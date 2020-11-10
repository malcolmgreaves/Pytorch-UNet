import os
from os.path import splitext
from os import listdir
from pathlib import Path
from typing import Iterator, Any, Mapping, Sequence, Dict, Optional, List

import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from torchvision import transforms


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=""):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, "Scale must be between 0 and 1"

        self.ids = [
            splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith(".")
        ]
        logging.info(f"Creating dataset with {len(self.ids)} examples")

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, "Scale is too small"
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + ".*")
        img_file = glob(self.imgs_dir + idx + ".*")

        assert (
            len(mask_file) == 1
        ), f"Either no mask or multiple masks found for the ID {idx}: {mask_file}"
        assert (
            len(img_file) == 1
        ), f"Either no image or multiple images found for the ID {idx}: {img_file}"
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert (
            img.size == mask.size
        ), f"Image and mask {idx} should be the same size, but are {img.size} and {mask.size}"

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            "image": torch.from_numpy(img).type(torch.FloatTensor),
            "mask": torch.from_numpy(mask).type(torch.FloatTensor),
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix="_mask")


""" TODO -- replace w/ import from other project """


def subtask_directories(embeding_dir: Path) -> Iterator[Path]:
    for _sd in os.listdir(str(embeding_dir)):
        subtask_dir = embeding_dir / _sd
        if not subtask_dir.is_dir():
            continue
        yield subtask_dir


def page_dirs(subtask_dir: Path) -> Iterator[Path]:
    for _pd in os.listdir(str(subtask_dir)):
        page_dir = subtask_dir / _pd
        if not page_dir.is_dir():
            continue
        yield page_dir


def load_page_image_from_dir(
    is_grayscale: bool, device: torch.device, page_dir: Path
) -> Image:
    raw_dict_p = str(page_dir / "raw.dict")
    try:
        with open(raw_dict_p, "rb") as rb:
            raw_dict: Mapping[str, Any] = torch.load(rb, map_location=device)
        page_t: torch.Tensor = raw_dict["page"]
        return transforms.ToPILImage(mode="L" if is_grayscale else "RGB")(page_t)

    except Exception as e:
        raise IOError(f"Could not read page images from raw.dict ({raw_dict_p})", e)


# def make_tiles(self.n_tiles, p)


class ReconstructDataset(Dataset):
    def __init__(
        self,
        embeding_dir: Path,
        device: torch.device,
        scale: float = 1.0,
        cache_in_memory: bool = False,
        n_tiles: Optional[int] = None,
        is_grayscale: bool = True,
    ) -> None:
        self.embeding_dir = embeding_dir
        self.scale = scale
        assert 0 < scale <= 1, "Scale must be between 0 and 1"
        self.n_tiles = n_tiles
        if self.n_tiles is not None and self.n_tiles <= 1:
            raise ValueError(f"If provided, # tiles must be > 1, not {self.n_tiles}")
        self.is_grayscale = is_grayscale

        self.subtask_dirs = list(subtask_directories(embeding_dir))
        self.device = device
        self.page_paths: Sequence[Path] = [
            page_dir
            for subtask_dir in subtask_directories(embeding_dir)
            for page_dir in page_dirs(subtask_dir)
        ]
        # self.index_to_img = []
        # if self.n_tiles:
        #     for i, p in enumerate(self.page_paths):
        #         tiles_of_p = make_tiles(self.n_tiles, p)
        #         for t in tiles_of_p:
        #             self.index_to_img.append(t)
        logging.info(f"Creating dataset with {len(self)} examples")
        self.cache_in_memory = cache_in_memory
        self._cache: List[Optional[Mapping[str, torch.FloatTensor]]] = (
            [None] * len(self) if self.cache_in_memory else []
        )

    def __len__(self):
        n = len(self.page_paths)
        return n * self.n_tiles if self.n_tiles else n

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, f"Scale is too small: {newW=}, {newH=}"
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        # normalize [0,255] to [0,1]
        if img_trans.max() > 1:  # type: ignore
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, index: int) -> Mapping[str, torch.FloatTensor]:
        """Loads image at the page directory corresponding to `index`, accessible under the "image" key.

        If `self.cache_in_memory` is true, then all images are eventually cached in-memory as they're accessed.
        """
        if self.cache_in_memory and self._cache[index] is not None:
            return self._cache[index]
        else:
            data = {"image": self.load_image_tensor(index)}
            if self.cache_in_memory:
                self._cache[index] = data
            return data  # type: ignore

    def load_image_tensor(self, index: int) -> torch.FloatTensor:
        """Load the page image that corresponds to `index`.
        """
        page_dir = self.page_paths[index]
        img = load_page_image_from_dir(
            is_grayscale=self.is_grayscale, device=self.device, page_dir=page_dir
        )
        img = self.preprocess(img, self.scale)
        return img
