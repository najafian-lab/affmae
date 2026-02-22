import torch
from src.data import create_dataloader
from src.config import load_config

@torch.no_grad()
def estimate_mean_std(dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    n_channels = None
    n_samples = 0
    mean = 0.0
    var = 0.0

    for images, *_ in dataloader:
        print('batch!')
        images = images.to(device, dtype=torch.float32)
        if n_channels is None:
            n_channels = images.shape[1]

        # flatten batch and spatial dimensions
        batch_samples = images.size(0)
        pixels = images.view(batch_samples, n_channels, -1)
        batch_mean = pixels.mean(dim=(0, 2))
        batch_var = pixels.var(dim=(0, 2), unbiased=False)

        # incremental update (avoids storing all data)
        mean = (n_samples * mean + batch_samples * batch_mean) / (n_samples + batch_samples)
        var = (n_samples * var + batch_samples * batch_var) / (n_samples + batch_samples)
        n_samples += batch_samples

    std = torch.sqrt(var)
    return mean.cpu(), std.cpu()

config = load_config('config/aff_small.yaml')
print('CONFIG', config)
dl = create_dataloader(config)
print(estimate_mean_std(dl))
