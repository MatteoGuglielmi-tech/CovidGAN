"""Discriminator model."""
from typing import List

import torch
import torch.nn as nn
from torchinfo import summary

class Discriminator(nn.Module):
    """Discminator model for ralsgan."""

    def __init__(self, opts):
        """Initialize Discriminator class.
            
        @param opts: (argparse.Namespace)
            Command-line arguments.
        """
        super(Discriminator, self).__init__()

        self.opts = opts

        def convlayer(
                n_input: int,
                n_output: int,
                k_size: int=4,
                stride: int=2,
                padding: int=0,
                bn=False
        ) -> List:
            """Convolutional layer block.

            @param n_input (int):
                Number of input channels.
            @param n_output (int):
                Number of output channels.
            @param k_size (int, optional): 
                Kernel size.
            @param stride (int, optional): 
                Stride.
            @param padding (int, optional):
                Padding.
            @param bn (bool, optional):
                Batch normalization.

            @return (List):
                Convolutional layer block.
            """
            block = []
            block.append(
                    nn.Conv2d(
                    in_channels=n_input, 
                    out_channels=n_output, 
                    kernel_size=k_size, 
                    stride=stride, 
                    padding=padding, 
                    bias=False
                    )
                )

            if bn:
                block.append(nn.BatchNorm2d(num_features=n_output))

            block.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

            return block
        
        self.backbone = nn.Sequential(
            *convlayer(n_input=self.opts.channels, n_output=32, k_size=4, stride=2, padding=1),
            *convlayer(n_input=32, n_output=64, k_size=4, stride=2, padding=1),
            *convlayer(n_input=64, n_output=128, k_size=4, stride=2, padding=1, bn=True),
            *convlayer(n_input=128, n_output=256, k_size=4, stride=2, padding=1, bn=True),
            *convlayer(n_input=256, n_output=512, k_size=4, stride=2, padding=1, bn=True),
            *convlayer(n_input=512, n_output=1024, k_size=4, stride=2, padding=1, bn=True),
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            )
    
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Forward pass of Discriminator class.

        @param imgs (torch.Tensor): 
            Input images.

        @return (torch.Tensor):
            Output critic value.
        """
        critic_value = self.backbone(imgs)
        critic_value = critic_value.view(imgs.shape[0], -1)
        return critic_value

    def _verbose(self) -> None:
        """Print summary of Discriminator class."""
        summary(model=self, input_size=(1, self.opts.channels, self.opts.img_size, self.opts.img_size))
