"""Original Discriminator from RaLSGAN paper."""
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from torchinfo import summary
# Spectral normalization stabilizes the training of discriminators (critics) in Generative Adversarial Networks (GANs)

class RaLSGANDiscriminator(nn.Module):
    """Improved Discriminator from RaLSGAN paper."""

    def __init__(self, img_size: int=256, channels: int=3, hidden_dim: int=128, n_gpus: int=1) -> None:
        """Initialize Discriminator.
        
        @param img_size (int, optional):
            Image size. Defaults to (256,256).
        @param channels (int, optional):
            Number of channels. Defaults to 3.
        @param hidden_dim (int, optional):
            Number of Discriminator hidden nodes. Defaults to 128.
        """
        super(RaLSGANDiscriminator, self).__init__()

        back = nn.Sequential()

        # 1st layer
        # input is (channels) x img_size x img_size
        back.add_module('InputSpectralConv2d', spectral_norm(nn.Conv2d(in_channels=channels, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1, bias=False)))
        back.add_module('InputSELU', nn.SELU(inplace=True))

        new_size = img_size // 2

        # middle layers
        mult = 1
        i = 0
        while new_size > 4:
            back.add_module('MiddleSpectralConv2d {}'.format(i), spectral_norm(nn.Conv2d(in_channels=hidden_dim * mult, out_channels=hidden_dim * (mult*2), kernel_size=4, stride=2, padding=1, bias=False)))
            back.add_module('MiddleSELU {}'.format(i), nn.SELU(inplace=True))
            mult *= 2
            new_size = new_size // 2
            i += 1

        # final layer
        # input is (hidden_dim * mult) x 4 x 4
        back.add_module('FinalSpectralConv2d', spectral_norm(nn.Conv2d(in_channels=hidden_dim * mult, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False)))

        # final size is 1 x 1 x 1

        self.backbone = back
        self.gpus = n_gpus
        self.channels = channels
        self.img_size = img_size

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        @param imgs (torch.Tensor):
            Input images. Shape (batch_size, channels, img_size, img_size).

        @return (torch.Tensor):
            Prediction. Shape (batch_size, 1).
        """
        # if imgs.is_cuda and self.gpus > 1:
        #     prediction = nn.parallel.data_parallel(self.backbone, imgs, range(self.gpus))
        # else:
        #     prediction = self.backbone(imgs)

        prediction = self.backbone(imgs)

        return prediction.view(-1)

    def _verbose(self) -> None:
        """Print summary of Discriminator class."""
        summary(model=self, input_size=(1, self.channels, self.img_size, self.img_size))
