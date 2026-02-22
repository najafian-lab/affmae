import os
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import v2
from torchvision import tv_tensors
import cv2
import tifffile
import albumentations.augmentations as AA
from src.transforms import ElasticTransform

IMAGE_MEAN = 0.6266
IMAGE_STD = 0.2259
THRESHOLD = 10

class EMDatasetMultiClass(Dataset):
    def __init__(
        self,
        base_path: str,
        test_dataset: bool = False,
        img_size: int = 512,
        apply_transforms: bool = False,
        indices: List[int] = [0, 2, 3],
        input_ext: str = ".tif",
        return_path: bool = True,
        perc_data: float = 1.0,
    ):
        self.test_dataset = test_dataset
        self.img_size = img_size
        self.apply_transforms = apply_transforms
        self.indices = indices
        self.return_path = return_path

        # Logic to determine folders
        train_place_in = os.path.join(base_path, "train")
        test_place_in = os.path.join(base_path, "val")

        if test_dataset:
            input_folder = os.path.join(test_place_in, "images")
            output_folder = os.path.join(test_place_in, "masks")
        else:
            input_folder = os.path.join(train_place_in, "images")
            output_folder = os.path.join(train_place_in, "masks")

        self.image_paths, self.mask_paths = self._load_image_names(input_folder, output_folder, input_ext)
        num_data_samples = int(len(self.image_paths) * perc_data)
        self.image_paths, self.mask_paths = self.image_paths[:num_data_samples], self.mask_paths[:num_data_samples]

        self.img_mean = torch.tensor(IMAGE_MEAN, dtype=torch.float)
        self.img_std = torch.tensor(IMAGE_STD, dtype=torch.float)

        self._setup_transforms()

    @staticmethod
    def _load_image_names(input_folder: str, output_folder: str, input_ext: str = ".tif") -> Tuple[np.ndarray, np.ndarray]:
        all_img = np.array(os.listdir(input_folder))
        all_tiff = np.array(os.listdir(output_folder))
        split_func = np.vectorize(lambda x: x.split(".")[0])
        
        # intersection of filenames
        img_paths = np.intersect1d(split_func(all_img), split_func(all_tiff))

        append_input = np.vectorize(lambda x: os.path.join(input_folder, x + input_ext))
        input_imgs = append_input(img_paths)

        append_output = np.vectorize(lambda x: os.path.join(output_folder, x + ".tiff"))
        output_imgs = append_output(img_paths)

        return input_imgs, output_imgs

    def _setup_transforms(self):
        base_transforms = [
            v2.ToImage(),
            v2.ConvertImageDtype(torch.float32),
            v2.Resize(size=(self.img_size, self.img_size), antialias=True, interpolation=v2.InterpolationMode.BILINEAR),
        ]

        if self.test_dataset or not self.apply_transforms:
            self.transforms = v2.Compose(base_transforms)
        else:
            augmentation_transforms = [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomApply([
                    v2.RandomAffine(degrees=45, translate=(0.0, 0.1), scale=(0.85, 1.15), shear=15.0)
                ], p=0.2),
                v2.RandomPhotometricDistort(
                    brightness=(1 - 0.3, 1 + 0.3), 
                    contrast=(1 - 0.3, 1 + 0.3),
                    saturation=(1,1), 
                    hue=(0,0),
                    p=0.6
                ),
                ElasticTransform(alpha=(0, 100), sigma=(10, 14), p=0.4),
            ]
            self.transforms = v2.Compose(base_transforms + augmentation_transforms)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_segm_image(self, id: int) -> np.ndarray:
        img = tifffile.imread(self.image_paths[id])
        return img if len(img.shape) == 3 else img.reshape(img.shape + (1,))

    def _load_segm_target(self, id: int) -> np.ndarray:
        img = tifffile.imread(self.mask_paths[id])
        if len(img.shape) == 4: img = img[:, :, :, 0]
        return img if len(img.shape) == 2 else img[[self.indices] if isinstance(self.indices, int) else self.indices, :, :]

    def __getitem__(self, index: int):
        image_np = self._load_segm_image(index)
        target_np_multi_channel = self._load_segm_target(index) 

        # CLAHE preprocessing
        clahe = cv2.createCLAHE(clipLimit=4.25, tileGridSize=(8, 8))
        try:
            image_np = clahe.apply(image_np)
        except:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            image_np = clahe.apply(image_np)
            
        if len(image_np.shape) == 2:
            image_np = image_np.reshape(image_np.shape + (1,))

        # create single channel class mask from multi-channel input
        C, H, W = target_np_multi_channel.shape
        target_np_multi_class = np.zeros((H, W), dtype=np.int64)

        for c in range(C):
            class_id = c + 1 # 0 is background
            is_active = target_np_multi_channel[c, :, :] > THRESHOLD
            target_np_multi_class[is_active] = class_id

        image = image_np
        target = tv_tensors.Mask(target_np_multi_class)

        image, target = self.transforms(image, target)
        image = image.float().sub_(self.img_mean).div_(self.img_std)

        if self.return_path:
            return image, target, (self.image_paths[index], self.mask_paths[index])
        return image, target


class EMTestDatasetMultiClass(Dataset):
    def __init__(
        self,
        base_path: str,
        img_size: int = 512,
        indices: List[int] = [0, 2, 3],
        input_ext: str = ".tif",
        return_path: bool = True,
    ):
        self.img_size = img_size
        self.indices = indices
        self.return_path = return_path

        # Logic to determine folders
        test_place_in = os.path.join(base_path, "test")

        input_folder = os.path.join(test_place_in, "images")
        output_folder = os.path.join(test_place_in, "masks")

        self.image_paths, self.mask_paths = self._load_image_names(input_folder, output_folder, input_ext)

        self.img_mean = torch.tensor(IMAGE_MEAN, dtype=torch.float)
        self.img_std = torch.tensor(IMAGE_STD, dtype=torch.float)

        self._setup_transforms()

    @staticmethod
    def _load_image_names(input_folder: str, output_folder: str, input_ext: str = ".tif") -> Tuple[np.ndarray, np.ndarray]:
        all_img = np.array(os.listdir(input_folder))
        all_tiff = np.array(os.listdir(output_folder))
        split_func = np.vectorize(lambda x: x.split(".")[0])
        
        # intersection of filenames
        img_paths = np.intersect1d(split_func(all_img), split_func(all_tiff))

        append_input = np.vectorize(lambda x: os.path.join(input_folder, x + input_ext))
        input_imgs = append_input(img_paths)

        append_output = np.vectorize(lambda x: os.path.join(output_folder, x + ".tiff"))
        output_imgs = append_output(img_paths)

        return input_imgs, output_imgs

    def _setup_transforms(self):
        base_transforms = [
            v2.ToImage(),
            v2.ConvertImageDtype(torch.float32),
            v2.Resize(size=(self.img_size, self.img_size), antialias=True, interpolation=v2.InterpolationMode.BILINEAR),
        ]
        self.transforms = v2.Compose(base_transforms)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_segm_image(self, id: int) -> np.ndarray:
        img = tifffile.imread(self.image_paths[id])
        return img if len(img.shape) == 3 else img.reshape(img.shape + (1,))

    def _load_segm_target(self, id: int) -> np.ndarray:
        img = tifffile.imread(self.mask_paths[id])
        if len(img.shape) == 4: img = img[:, :, :, 0]
        return img if len(img.shape) == 2 else img[[self.indices] if isinstance(self.indices, int) else self.indices, :, :]

    def __getitem__(self, index: int):
        image_np = self._load_segm_image(index)
        target_np_multi_channel = self._load_segm_target(index) 

        # CLAHE preprocessing
        clahe = cv2.createCLAHE(clipLimit=4.25, tileGridSize=(8, 8))
        try:
            image_np = clahe.apply(image_np)
        except:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            image_np = clahe.apply(image_np)
        
        if len(image_np.shape) == 2:
            image_np = image_np.reshape(image_np.shape + (1,))

        # create single channel class mask from multi-channel input
        C, H, W = target_np_multi_channel.shape
        target_np_multi_class = np.zeros((H, W), dtype=np.int64)

        for c in range(C):
            class_id = c + 1 # 0 is background
            is_active = target_np_multi_channel[c, :, :] > THRESHOLD
            target_np_multi_class[is_active] = class_id

        image = image_np
        target = tv_tensors.Mask(target_np_multi_class)

        image, target = self.transforms(image, target)
        image = image.float().sub_(self.img_mean).div_(self.img_std)

        if self.return_path:
            return image, target, (self.image_paths[index], self.mask_paths[index])
        return image, target


def build_finetune_dataloader(cfg, is_train: bool):
    dataset = EMDatasetMultiClass(
        base_path=cfg.base_path,
        test_dataset=not is_train,
        img_size=cfg.img_size,
        indices=cfg.indices,
        input_ext=cfg.input_ext,
        apply_transforms=is_train
    )
    print(f"DATASET LEN (IS_TRAIN: {is_train})")
    print(len(dataset))

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size if is_train else 1,
        shuffle=is_train,
        num_workers=cfg.num_workers,
        pin_memory=False,
        persistent_workers=True if cfg.num_workers > 0 else False
    )

def build_finetune_dataloader_perc(cfg, data_perc, is_train: bool):
    dataset = EMDatasetMultiClass(
        base_path=cfg.base_path,
        test_dataset=not is_train,
        img_size=cfg.img_size,
        indices=cfg.indices,
        input_ext=cfg.input_ext,
        apply_transforms=is_train,
        perc_data=data_perc
    )
    print(f"DATASET LEN (IS_TRAIN: {is_train})")
    print(len(dataset))

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size if is_train else 1,
        shuffle=is_train,
        num_workers=cfg.num_workers,
        pin_memory=False,
        persistent_workers=True if cfg.num_workers > 0 else False
    )

# used for calculating class weights
def build_util_dataloader(data_path, img_size):
    dataset = EMDatasetMultiClass(
        base_path=data_path,
        test_dataset=False,
        img_size=img_size,
        indices=[0, 1, 2],
        input_ext='.tiff',
        apply_transforms=False
    )

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )

def build_test_dataloader(cfg):
    dataset = EMTestDatasetMultiClass(
        base_path=cfg.base_path,
        img_size=cfg.img_size,
        indices=cfg.indices,
        input_ext=cfg.input_ext,
    )
    print(f"TEST DATASET LEN: {len(dataset)}")

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.num_workers > 0 else False
    )