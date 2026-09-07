"""Dataset normalization statistics, with no dependencies. """

__all__ = ["IMAGE_MEAN", "IMAGE_STD", "PRETRAIN_IMAGE_MEAN",
           "PRETRAIN_IMAGE_STD"]

#: Labelled FPW split, used by the finetune loader and by inference.
IMAGE_MEAN = 0.6266
IMAGE_STD = 0.2259

#: Unlabelled pretraining corpus.
PRETRAIN_IMAGE_MEAN = 0.5562
PRETRAIN_IMAGE_STD = 0.2396
