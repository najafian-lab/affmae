import torch
import webdataset as wds
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import random
import os
from typing import Dict
from pathlib import Path
import torch.distributed as dist

# our dataset has exactly that number of samples
TOTAL_SAMPLES = 187270


def random_transform(image):
    if len(image.shape) == 3:
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Randomly apply a transformation to the image
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
    if random.random() < 0.5:
        image = cv2.flip(image, 0)
    
    # d 1-40 (int)
    # o_color 0.1-6.5
    # o_space 0.1-25.0
    # clip_limit 0.5-3
    # tile_size 1-6 (int)
    # contrast = 1-3.5

    d = random.randint(3, 40)
    sigma_color = random.uniform(0.5, 6.5)
    sigma_space = random.uniform(1.0, 25.0)
    clip_limit = random.uniform(0.5, 3)
    tile_size = random.randint(2, 6)
    contrast = random.uniform(1, 3.5)

    # random blur
    if random.random() < 0.9:
        image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    # random CLAHE
    if random.random() < 0.9:
        image = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size)).apply(image)

    # random contrast
    if random.random() < 0.25:
        image = (image.astype(np.float32) - np.mean(image)) * contrast + np.mean(image)
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def apply_custom_processing(sample: Dict) -> Dict:
    """
    applies CLAHE and removes the bottom white bar from an image.
    """
    # webDataset's .decode("pil") puts the decoded image in the key matching its extension
    try:
        pil_image = sample[list(sample.keys())[2]]
    except KeyError as exc:
        print(sample.keys())
        raise exc

    pil_image = pil_image.convert("L")
    img_np = np.array(pil_image)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    img_np = clahe.apply(img_np) # currently image has no channel dim

    # remove the white text bar
    height = img_np.shape[0]
    h_60 = int(height * 0.6)
    white_pixel_count = np.sum(img_np[h_60:] == 255, axis=1)
    row_percentages = white_pixel_count / img_np.shape[1]
    over_90_percent_mask = row_percentages > 0.95

    if np.any(over_90_percent_mask):
        first_row_to_remove = np.argmax(over_90_percent_mask)
        cropped_img_array = img_np[:h_60 + first_row_to_remove]
    else:
        cropped_img_array = img_np  

    # convert back to a PIL Image and update the sample dictionary
    sample["png"] = Image.fromarray(cropped_img_array)

    return sample


def create_transforms(img_size: int, n_channels: int) -> transforms.Compose:
    transform_list = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]
    if n_channels == 1:
        transform_list.append(transforms.Normalize((0.5562,), (0.2396,)))
    else:
        transform_list.append(transforms.Normalize((0.5562, 0.5562, 0.5562), (0.2396, 0.2396, 0.2396)))
    return transforms.Compose(transform_list)

def build_pretrain_dataloader(config) -> DataLoader:
    
    transform = create_transforms(config.img_size, config.in_channels)

    urls = config.path
    dataset = wds.WebDataset(urls, shardshuffle=True)
    batch_size = config.batch_size

    dataset = (
        dataset
        .shuffle(1000)
        .decode("pil")
        .map(apply_custom_processing) 
        .map_dict(png=transform)
        .map(lambda sample: (sample["png"], 0)) # return (image, dummy_label)
    )

    return DataLoader(
        dataset.batched(batch_size), 
        batch_size=None, 
        num_workers=config.num_workers, 
        prefetch_factor=config.prefetch_factor,
        pin_memory=config.pin_memory
    )

def get_stable_visualization_batch(config, device):
    # this is deterministic
    dataset = wds.WebDataset(config.path, shardshuffle=False)

    transform = create_transforms(config.img_size, config.in_channels)

    dataset = (
        dataset
        .decode("pil")
        .map(apply_custom_processing) 
        .map_dict(png=transform)
        .map(lambda sample: (sample["png"], 0))
    )

    # 0 workers
    loader = DataLoader(
        dataset.batched(config.batch_size), 
        batch_size=None, 
        num_workers=0 
    )

    vis_images, _ = next(iter(loader))
    return vis_images.to(device)