"""Original Generator from RaLSGAN paper."""
import torch
import torch.nn as nn
from torchinfo import summary
# https://pytorch.org/docs/stable/generated/torch.nn.utils.spectral_norm.html

class RaLSGANGenerator(nn.Module):
    """Improved Generator from RaLSGAN paper."""

    def __init__(self, img_size: int=256, latent_dim: int=128, channels: int=3, hidden_dim: int=128, n_gpus: int=1) -> None:
        """Initialize Generator.
        
        @param img_size (int, optional):
            Image size. Defaults to (256,256).
        @param latent_dim (int, optional):
            Latent dimension. Defaults to 128.
        @param channels (int, optional):
            Number of channels. Defaults to 3.
        @param hidden_dim (int, optional):
            Number of Generator hidden nodes. Defaults to 128.
        """
        super(RaLSGANGenerator, self).__init__()

        self.gpus = n_gpus
        self.latent_dim = latent_dim

        back = nn.Sequential()

        # original
        mult = img_size // 8 # gives how many layers to use

        # shallower version
        # mult = img_size // 16

        # 1st layer
        back.add_module('StartConvTranspose2d', nn.ConvTranspose2d(in_channels=latent_dim, out_channels=hidden_dim * mult, kernel_size=4, stride=1, padding=0, bias=False))
        # imporoves stability. It replaces BatchNorm with RELU.
        back.add_module('StartSELU', nn.SELU(inplace=True))

        # middle layers
        i=0
        while mult > 1:
            back.add_module('MiddleUpSample {}'.format(i), nn.Upsample(scale_factor=2, mode='nearest'))
            back.add_module('Middle-Conv2d {}'.format(i), nn.Conv2d(hidden_dim * mult, hidden_dim * (mult//2), kernel_size=3, stride=1, padding=1))
            back.add_module('MiddleSELU {}'.format(i), nn.SELU(inplace=True))
            mult = mult // 2
            i += 1
        
        # final layer
        back.add_module('FinalUpsample', nn.Upsample(scale_factor=2, mode='nearest'))
        back.add_module('FinalConv2d', nn.Conv2d(in_channels=hidden_dim, out_channels=channels, kernel_size=3, stride=1, padding=1))

        back.add_module('Tanh', nn.Tanh())


        self.backbone = back

    def forward(self, z) -> torch.Tensor:
        """Forward pass.
        
        @param z (torch.Tensor):
            Input noise. Shape (batch_size, latent_dim, 1, 1).

        @return (torch.Tensor):
            Generated image. Shape (batch_size, channels, img_size, img_size).
        """
        # if z.is_cuda and self.gpus > 1:
        #     output = nn.parallel.data_parallel(self.backbone, input, range(self.gpus))
        # else:
        #     output = self.backbone(z)

        output = self.backbone(z)

        return output

    def _verbose(self) -> None:
        """Print summary of Discriminator class."""
        summary(model=self, input_size=(1, self.latent_dim, 1, 1))
    
    def save_weights(self, path: str) -> None:
        """Save weights of Generator class.

        @param path (str):
            Path to save weights.
        """
        torch.save(obj=self.state_dict(), f=path)

    def load_weights(self, path: str) -> None:
        """Load weights of Generator class.

        @param path (str):
            Path to load weights.
        """
        self.load_state_dict(state_dict=torch.load(f=path))
