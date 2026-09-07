import webdataset as wds
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import random
from typing import Dict

# Pretraining-corpus statistics. Named because they were literals in three
# places, and the renderers must denormalize with the same numbers the loader
# normalized with -- the finetune split has different ones (IMAGE_MEAN /
# IMAGE_STD in affmae/data/finetune_dataset.py).
from affmae.data.stats import (  # noqa: F401  (re-export)
    PRETRAIN_IMAGE_MEAN,
    PRETRAIN_IMAGE_STD,
)

#: Sample count of the EM dataset the paper pretrains on. Only a default: a
#: WebDataset is an IterableDataset with no ``__len__``, so the epoch length has
#: to be stated rather than measured. Override it with ``data.total_samples`` in
#: the config when pretraining on your own shards -- leaving it at this value
#: silently truncates a smaller dataset's epoch or repeats a larger one's.
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


def create_transforms(img_size: int, n_channels: int,
                      normalize: bool = True) -> transforms.Compose:
    transform_list = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]
    # normalize=False is what a statistics pass needs: measuring mean/std through
    # a pipeline that already normalized reports the residual (~0, ~1), not the
    # dataset's statistics.
    if normalize:
        mean, std = PRETRAIN_IMAGE_MEAN, PRETRAIN_IMAGE_STD
        transform_list.append(
            transforms.Normalize((mean,) * n_channels, (std,) * n_channels))
    return transforms.Compose(transform_list)

def build_pretrain_dataloader(config, world_size: int = 1,
                              normalize: bool = True):
    """Build the pretraining loader, sharded across ranks when distributed.

    WebDataset is an IterableDataset, so ``DistributedSampler`` does not apply.
    Three things have to be handled explicitly:

    1. ``nodesplitter=wds.split_by_node`` gives each rank a disjoint set of
       shards. Without it every rank decodes the whole dataset, so ranks train
       on identical batches and the all-reduce just averages duplicates.
    2. ``config.batch_size`` is the **global** batch, divided by ``world_size``
       to get the per-rank micro-batch, so the effective batch is independent of
       how many GPUs are used.
    3. The epoch length is pinned to ``batches_per_rank``. Shard counts and
       per-shard sample counts are uneven, so without this the first rank to run
       dry leaves the others waiting in the next all-reduce until NCCL's
       watchdog aborts the job.

    ``with_epoch`` goes on the loader, not the dataset: under a multi-worker
    DataLoader each worker builds its own pipeline with its own budget, which
    would multiply batches per rank by ``num_workers``.

    Args:
        config: Config, needs ``path``, ``batch_size``, ``img_size``,
            ``in_channels``, ``num_workers``, ``prefetch_factor``,
            ``pin_memory``. Optional ``total_samples`` sets the epoch length,
            defaulting to :data:`TOTAL_SAMPLES`.
        world_size: int, number of ranks. 1 gives the single-process path.
    Returns:
        (loader, batches_per_epoch). ``batches_per_epoch`` is per rank.
    """
    transform = create_transforms(config.img_size, config.in_channels,
                                  normalize=normalize)
    per_rank_batch_size = max(1, config.batch_size // world_size)

    distributed = world_size > 1
    # shardshuffle must be an int in current webdataset; True is deprecated.
    dataset = wds.WebDataset(
        config.path,
        shardshuffle=100,
        nodesplitter=wds.split_by_node if distributed else None,
    )

    dataset = (
        dataset
        .shuffle(1000)
        .decode("pil")
        .map(apply_custom_processing)
        .map_dict(png=transform)
        .map(lambda sample: (sample["png"], 0))  # return (image, dummy_label)
        .batched(per_rank_batch_size)
    )

    total_samples = int(getattr(config, "total_samples", None) or TOTAL_SAMPLES)
    batches_per_rank = (total_samples // world_size) // per_rank_batch_size
    if batches_per_rank < 1:
        raise ValueError(
            f"total_samples={total_samples} with global batch_size="
            f"{config.batch_size} over {world_size} rank(s) leaves "
            f"{batches_per_rank} batches per epoch. Set data.total_samples to "
            f"your dataset's sample count, or lower the batch size.")

    if not distributed:
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            pin_memory=config.pin_memory,
        ), batches_per_rank

    loader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=config.pin_memory,
    )
    return (loader.with_epoch(nbatches=batches_per_rank)
                  .with_length(batches_per_rank), batches_per_rank)

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