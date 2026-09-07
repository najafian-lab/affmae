"""The single image-preprocessing chain, shared by the datasets and inference.
The current implementation is:
1. CLAHE (``clipLimit=4.25``, ``tileGridSize=(8, 8)``) -- local contrast
   equalization; electron micrographs vary a lot in exposure.
2. Reshape to a single channel.
3. Bilinear resize to ``img_size`` with antialiasing.
4. Normalize by the dataset mean/std.
"""

from typing import Optional

import numpy as np
import torch

__all__ = [
    "CLAHE_CLIP_LIMIT",
    "CLAHE_TILE_GRID",
    "MASK_THRESHOLD",
    "apply_clahe",
    "load_image",
    "preprocess_image",
    "multichannel_mask_to_labels",
]

# Matches the datasets exactly; changing either changes what the model sees.
CLAHE_CLIP_LIMIT = 4.25
CLAHE_TILE_GRID = (8, 8)

# A channel of a multi-channel mask counts as that class above this value.
MASK_THRESHOLD = 10


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE, converting to greyscale first if needed.

    Args:
        image: [H, W] or [H, W, 3] uint8 array.
    Returns:
        [H, W] uint8 array.
    """
    import cv2

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                            tileGridSize=CLAHE_TILE_GRID)
    if image.ndim == 3 and image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.dtype != np.uint8:
        # CLAHE requires 8-bit input; scale by the observed range so 12/16-bit
        # micrographs are not clipped.
        lo, hi = float(image.min()), float(image.max())
        image = ((image - lo) / (hi - lo + 1e-8) * 255.0).astype(np.uint8)
    return clahe.apply(image)


def load_image(source) -> np.ndarray:
    """Read an image from whatever the caller has.

    Args:
        source: a path (str or PathLike; ``.tif``/``.tiff`` via tifffile,
            anything else via Pillow), a numpy array, a PIL image, or a tensor.
    Returns:
        [H, W] or [H, W, C] array.
    Raises:
        FileNotFoundError: if a given path does not exist.
    """
    import os

    if isinstance(source, torch.Tensor):
        array = source.detach().cpu().numpy()
        return array[0] if array.ndim == 3 and array.shape[0] in (1, 3) else array
    if isinstance(source, np.ndarray):
        return source
    if hasattr(source, "convert"):        # PIL.Image
        return np.array(source)

    path = os.fspath(source)
    if not os.path.exists(path):
        raise FileNotFoundError(f"image not found: {path}")
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile

        return tifffile.imread(path)
    from PIL import Image

    return np.array(Image.open(path))


def preprocess_image(source, img_size: int, mean: Optional[float] = None,
                     std: Optional[float] = None,
                     use_clahe: bool = True) -> torch.Tensor:
    """Turn an arbitrary image into a normalized model input.

    Args:
        source: anything :func:`load_image` accepts.
        img_size: side length the model expects; the image is resized square.
        mean: normalization mean, or None for the dataset default.
        std: normalization std, or None likewise.
        use_clahe: apply CLAHE. Leave True to match training.
    Returns:
        [1, 1, img_size, img_size] float tensor, normalized.
    """
    from torchvision.transforms import v2

    from .stats import IMAGE_MEAN, IMAGE_STD

    mean = IMAGE_MEAN if mean is None else mean
    std = IMAGE_STD if std is None else std

    array = load_image(source)
    if use_clahe:
        array = apply_clahe(array)
    if array.ndim == 2:
        array = array[..., None]

    # Mirror the dataset exactly: ToImage -> ConvertImageDtype(float32) ->
    # Resize -> normalize. ConvertImageDtype is what maps uint8 0..255 to 0..1;
    # omitting it makes every normalized value ~255x too large.
    transform = v2.Compose([
        v2.ToImage(),
        v2.ConvertImageDtype(torch.float32),
        v2.Resize(size=(img_size, img_size), antialias=True,
                  interpolation=v2.InterpolationMode.BILINEAR),
    ])
    tensor = transform(np.ascontiguousarray(array))
    return tensor.sub_(float(mean)).div_(float(std)).unsqueeze(0)


def multichannel_mask_to_labels(mask: np.ndarray,
                                threshold: int = MASK_THRESHOLD) -> np.ndarray:
    """Collapse a per-class mask stack into a label map.

    Args:
        mask: [C, H, W] array, one channel per foreground class.
        threshold: a channel counts as active above this value.
    Returns:
        [H, W] int64 labels, 0 = background, class c at channel c-1.
    """
    channels, height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int64)
    for channel in range(channels):
        labels[mask[channel] > threshold] = channel + 1
    return labels
