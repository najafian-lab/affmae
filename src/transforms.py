import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
import torch
from torchvision import tv_tensors

class ElasticTransform:
    """
    Elastic deformation of images based on implementation by MIC-DKFZ.
    
    Args:
        alpha (tuple or float): Scaling factor for deformation intensity. 
                                If tuple, a value is sampled uniformly from (min, max).
        sigma (tuple or float): Smoothing factor (Gaussian std dev). 
                                If tuple, a value is sampled uniformly from (min, max).
        p (float): Probability of applying the transform.
    """
    def __init__(self, alpha=(0, 200), sigma=(10, 13), p=0.5):
        self.alpha_range = alpha if isinstance(alpha, (tuple, list)) else (alpha, alpha)
        self.sigma_range = sigma if isinstance(sigma, (tuple, list)) else (sigma, sigma)
        self.p = p

    def __call__(self, img, target=None):
        """
        Args:
            img (Tensor or np.ndarray): Image of shape (C, H, W)
            target (Tensor or np.ndarray): Mask of shape (H, W) or (C, H, W)
        """
        if np.random.random() > self.p:
            return (img, target) if target is not None else img

        is_tensor = torch.is_tensor(img)
        if is_tensor:
            img_np = img.numpy()
            target_np = target.numpy() if target is not None else None
        else:
            img_np = img
            target_np = target

        shape = img_np.shape[1:] 
        
        alpha = np.random.uniform(self.alpha_range[0], self.alpha_range[1])
        sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])

        coords = self._generate_elastic_transform_coordinates(shape, alpha, sigma)

        coords = np.array(coords).reshape(len(shape), shape[0], shape[1])

        deformed_img = np.zeros_like(img_np)
        for c in range(img_np.shape[0]):
            deformed_img[c] = map_coordinates(img_np[c], coords, order=3, mode='reflect')

        deformed_target = None
        if target_np is not None:
            # Handle if target is (H, W) or (C, H, W)
            if len(target_np.shape) == 2:
                deformed_target = map_coordinates(target_np, coords, order=0, mode='reflect')
            else:
                deformed_target = np.zeros_like(target_np)
                for c in range(target_np.shape[0]):
                    deformed_target[c] = map_coordinates(target_np[c], coords, order=0, mode='reflect')

        if is_tensor:
            img_out = torch.from_numpy(deformed_img).float()
            if target is not None:
                # Keep target format consistent
                if isinstance(target, tv_tensors.Mask):
                    target_out = tv_tensors.Mask(torch.from_numpy(deformed_target))
                else:
                    target_out = torch.from_numpy(deformed_target)
        else:
            img_out = deformed_img
            target_out = deformed_target

        return (img_out, target_out) if target is not None else img_out

    @staticmethod
    def _generate_elastic_transform_coordinates(shape, alpha, sigma):
        """
        Adapted directly from MIC-DKFZ batchgenerators
        """
        n_dim = len(shape)
        offsets = []
        for _ in range(n_dim):
            random_state = np.random.random(shape) * 2 - 1
            offset = gaussian_filter(random_state, sigma, mode="constant", cval=0) * alpha
            offsets.append(offset)
        
        tmp = tuple([np.arange(i) for i in shape])
        coords = np.meshgrid(*tmp, indexing='ij')
        
        indices = [np.reshape(i + j, (-1, 1)) for i, j in zip(offsets, coords)]
        return indices