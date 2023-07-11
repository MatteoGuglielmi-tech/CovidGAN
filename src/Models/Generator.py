"""Generator model for GAN."""
from typing import List

import torch
import torch.nn as nn
import torchvision.utils as vutils
from torchinfo import summary


class GaussianNoiseLayer(nn.Module):
    """Gaussian noise regularizer."""

    def __init__(self, sigma: float=0.1, is_relative_detach: bool=True):
        """Initialize GaussianNoiseLayer class.

        @param sigma (float, optional):
            Standard deviation of noise distribution.
        @param is_relative_detach (bool, optional):
            Whether to detach noise from computational graph. In this case, noise amplitude is 
            not treated as parameter to be optmized in the optimization process.
        """
        super(GaussianNoiseLayer, self).__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).cuda()

    def forward(self, x) -> torch.Tensor:
        """Forward pass.
            
        @param x (torch.Tensor):
            Input tensor.

        @return (torch.Tensor):
            Noisy input tensor.
        """
        if self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x 


class Generator(nn.Module):
    """Generator model for GAN."""

    def __init__(self, opts, training: bool=True) -> None:
        """Initialize Generator class.
    
        @param opts (argparse.Namespace):
            Command-line arguments.
        @param training (bool, optional):
            Whether to use noise during training phase. Default: True.
        """
        super(Generator, self).__init__()

        self.opts = opts
        self.training = training

        def convlayer(
                n_input,
                n_output,
                k_size=4,
                stride=2,
                padding=0
        ) -> List:
            """Convolutional layer block.

            @param n_input (int):
                Number of input channels.
            @param n_output (int): 
                Number of output channels.
            @param k_size (int, optional):
                Kernel size. Default: 4.
            @param stride (int, optional):
                Stride. Default: 2.
            @param padding (int, optional):
                Padding. Default: 0.

            @return (List):
                Convolutional layer block.
            """
            block = [
                nn.ConvTranspose2d(
                    in_channels=n_input, 
                    out_channels=n_output, 
                    kernel_size=k_size, 
                    stride=stride, 
                    padding=padding, 
                    bias=False
                    ),
                nn.BatchNorm2d(num_features=n_output),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5)
            ]

            # I want noise only during training phase 
            # if self.training:
            #     block.append(GaussianNoiseLayer())
            
            return block

        self.backbone = nn.Sequential(
            *convlayer(n_input=self.opts.latent_dim, n_output=1024, k_size=4, stride=1, padding=0),
            *convlayer(n_input=1024, n_output=512, k_size=4, stride=2, padding=1),
            *convlayer(n_input=512, n_output=256, k_size=4, stride=2, padding=1),
            *convlayer(n_input=256, n_output=128, k_size=4, stride=2, padding=1),
            *convlayer(n_input=128, n_output=64, k_size=4, stride=2, padding=1),
            *convlayer(n_input=64, n_output=32, k_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels=32, out_channels=self.opts.channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
            )


    def forward(self, z) -> torch.Tensor:
        """Forward pass of Generator class.

        @param z (torch.Tensor):
            Input noise.
        """
        z = z.view(-1, self.opts.latent_dim, 1, 1)
        out = self.backbone(z)
        return out


    def _verbose(self) -> None:
        """Print summary of Discriminator class."""
        summary(model=self, input_size=(1, self.opts.latent_dim, 1, 1))
    
    
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


    def evaluate(self, samples2generate: int, expFolder: str) -> None:
        """Generate synthetic images.
        
        @param samples2generate (int):
            Number of synthetic images to generate.
        @param expFolder (str):
            Path to save synthetic images.

        @return (torch.Tensor):
            Synthetic images.
        """
        self.eval() 
        with torch.no_grad():
            for idx in range(samples2generate):
                synthetic = self.forward(z=torch.randn(1, self.opts.latent_dim, 1, 1).cuda())
                vutils.save_image(
                    tensor=synthetic.detach(),
                    fp=f"{expFolder}/synthetic_image-{idx + 1}.png",
                    # normalize=True,
                    value_range=(0, 255),
                    )
