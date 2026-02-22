import os
import random
import numpy as np
import torch
import cv2
import tifffile
import matplotlib.pyplot as plt
from typing import List, Tuple
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors

from src.transforms import ElasticTransform 


BASE_PATH = "/homes/iws/ziawang/Documents/lab/affmae_root/data/Paratubular Capillaries 8_19_25" 

IMG_SIZE = 512
INPUT_EXT = ".tiff"

IMAGE_MEAN = [0.0] 
IMAGE_STD = [1.0]
THRESHOLD = 0.5


class EMDatasetMultiClass(Dataset):
    def __init__(
        self,
        base_path: str,
        test_dataset: bool = False,
        img_size: int = 512,
        apply_transforms: bool = False,
        indices: List[int] = [0, 1, 2],
        input_ext: str = ".tif",
        return_path: bool = True,
    ):
        self.test_dataset = test_dataset
        self.img_size = img_size
        self.apply_transforms = apply_transforms
        self.indices = indices
        self.return_path = return_path

        # Logic to determine folders
        train_place_in = os.path.join(base_path, "train")
        test_place_in = os.path.join(base_path, "test")

        if test_dataset:
            input_folder = os.path.join(test_place_in, "images")
            output_folder = os.path.join(test_place_in, "masks")
        else:
            input_folder = os.path.join(train_place_in, "images")
            output_folder = os.path.join(train_place_in, "masks")
        
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"Input folder not found: {input_folder}")

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

        if self.test_dataset or not self.apply_transforms:
            self.transforms = v2.Compose(base_transforms)
        else:
            augmentation_transforms = [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomAdjustSharpness(sharpness_factor=0.8, p=0.25),
                v2.RandomAdjustSharpness(sharpness_factor=1.25, p=0.25),
                v2.RandomAffine(
                    degrees=30, 
                    translate=(0.0, 0.1), 
                    scale=(0.9, 1.3), 
                    shear=20.0
                ),
                ElasticTransform(alpha=(100, 100), sigma=(10, 10), p=0.6)
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
            if image_np.dtype != np.uint8:
                 norm_img = cv2.normalize(image_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                 image_np = clahe.apply(norm_img)
            else:
                 image_np = clahe.apply(image_np)
        except:
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            image_np = clahe.apply(image_np)

        if len(image_np.shape) == 2:
            image_np = image_np.reshape(image_np.shape + (1,))

        if len(target_np_multi_channel.shape) == 3:
            C, H, W = target_np_multi_channel.shape
            target_np_multi_class = np.zeros((H, W), dtype=np.int64)
            for c in range(C):
                class_id = c + 1 
                is_active = target_np_multi_channel[c, :, :] > THRESHOLD
                target_np_multi_class[is_active] = class_id
        else:
            target_np_multi_class = target_np_multi_channel

        image = image_np
        target = tv_tensors.Mask(target_np_multi_class)

        image, target = self.transforms(image, target)
        image = image.float().sub_(self.img_mean).div_(self.img_std)

        if self.return_path:
            return image, target, (self.image_paths[index], self.mask_paths[index])
        return image, target


def visualize_grid():
    print(f"Loading dataset from: {BASE_PATH}")
    try:
        dataset = EMDatasetMultiClass(
            base_path=BASE_PATH,
            test_dataset=False,
            img_size=IMG_SIZE,
            apply_transforms=True, 
            input_ext=INPUT_EXT,
            return_path=True
        )
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return

    print(f"Found {len(dataset)} images.")
    
    if len(dataset) < 25:
        print("Dataset is too small for a 5x5 grid, using all images.")
        indices = list(range(len(dataset)))
    else:
        indices = random.sample(range(len(dataset)), 25)

    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    fig.suptitle(f"5x5 Random Augmentation Grid", fontsize=16)

    print("Generating grid...")
    
    for idx, ax in zip(indices, axes.flat):
        img_tensor, mask_tensor, (img_path, mask_path) = dataset[idx]
    
        rel_path = os.path.relpath(img_path, BASE_PATH)
        
        # Un-normalize for display
        img = img_tensor.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        mean = np.array(IMAGE_MEAN)
        std = np.array(IMAGE_STD)
        img = (img * std) + mean
        img = np.clip(img, 0.0, 1.0)
        
        if img.shape[2] == 1:
            img = img.squeeze(2)
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
            
        # Set Title with relative path
        ax.set_title(rel_path, fontsize=6)
        ax.axis('off')

    plt.tight_layout()
    save_path = "/homes/iws/ziawang/Documents/lab/affmae_root/affmae/scripts/aug_viz.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=125)
    print(f"Saved visualization to: {save_path}")

if __name__ == "__main__":
    visualize_grid()