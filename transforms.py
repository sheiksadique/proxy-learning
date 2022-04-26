import torch


def expand_along_time(img: torch.Tensor, time_steps=50):
    """
    Extend an image along a time axis
    """
    img = img.unsqueeze(0)
    spks = torch.tile(img, [time_steps, 1, 1, 1])
    return spks


class ImageAndCurrent:
    """
    Expand current input image into an image and time streched image
    """

    def __init__(self, time_steps=50):
        self.time_steps = time_steps

    def __call__(self, img):
        return img, expand_along_time(img, time_steps=self.time_steps)
