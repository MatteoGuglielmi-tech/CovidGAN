"""Utilities for PyTorch."""
import datetime
import random
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as T
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from .parse import opts

def set_seed():
    """Set the random seed."""
    if not opts.random_seed:
        manualSeed = 999
    else:
        manualSeed = random.randint(1, 10000)
        print("Random Seed: ", manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

def writer_setup():
    """Set up the writer."""
    writer = SummaryWriter(log_dir=f"all_runs/{datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}")
    return writer

def get_normalization_data(loader: DataLoader) -> Tuple:
    """Return data and average for each channel to perform normalization.

    @params loader (DataLoader):
        dataloader for the images
    @returns tuple(mean,std): 
        mean for each channel and standard deviation for each channel

    @usage : this function is used to compute the mean and the standard deviation of the dataset and
    it is sufficient to run it once.
    """
    # https://stackoverflow.com/questions/60101240/finding-mean-and-standard-deviation-across-image-channels-pytorch/60803379#60803379
    mean = 0.0
    var = 0.0
    std = 0.0
    nr_imgs = 0

    with tqdm(iterable=loader, unit='batch') as pbar:
        for idx, (batch, _) in enumerate(pbar):
            pbar.set_description("Epoch [%d/%d]" % (idx + 1, len(loader))) 
            batch = batch.cuda()
            # [batch_size, channels, height * width]
            batch = batch.view(batch.size(dim=0), batch.size(1), -1)
            nr_imgs += batch.size(dim=0)
            mean += batch.mean(dim=2).sum(dim=0)
            var += batch.var(dim=2).sum(dim=0)

    mean = mean / nr_imgs
    var = var / nr_imgs
    std = torch.sqrt(var)

    print(f"Mean: {mean}")
    print(f"Std: {std}")

    return mean, std


def parse_data(
    root: str,
    img_size: Tuple,
) -> Dataset:
    """Return the dataloader for a score folder.

    @params root (str):
        path to the folder containing the images
    @params img_size (int):
        size of the images
    @params shuffle (bool):
        whether to shuffle the data or not. Defaults to True.

    @returns dataset (class torchvision.datasets.ImageFolder):
        dataset for the images
    """
    transform = list()
    transform.append(T.Resize((img_size, img_size)))

    if opts.channels == 1:
        transform.append(T.Grayscale())

    # pixels' value in range [0, 1.0]
    transform.append(T.ToTensor())

    if opts.full_scale:
        transform.append(T.Lambda(lambda img: img*255.))

    if opts.normalize:
        if opts.standard_normalization:
            print("Using standard normalization...")
            # using standard normalization images are in range [-1.0, 1.0]
            if opts.channels == 1:
                transform.append(T.Normalize((0.5), (0.5)))
            else:
                transform.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        # else: 
        #     print("Using computed normalization...")
        #     print("Is it a new dataset? (y/n)")
        #     if input() == "y":
        #         print("Computing normalization...")
        #         if opts.channels == 1:
        #             tmp_transform = T.Compose([T.Resize((img_size, img_size)),T.Grayscale(),T.ToTensor()])
        #         else:
        #             tmp_transform = T.Compose([T.Resize((img_size, img_size)),T.ToTensor()])
        #         tmp_dataset = ImageFolder(root=root, transform=tmp_transform)
        #         tmp_dataloader = DataLoader(dataset=tmp_dataset, batch_size=48, shuffle=True)
        #         mean, std = get_normalization_data(loader=tmp_dataloader)
        #     else:
        #         print("Using stored normalization...")
        #         mean = (0.0460, 0.0464, 0.0516)
        #         std = (0.1019, 0.1028, 0.1163)
        else:
            print("Using stored normalization...")
            if opts.channels == 1:
                mean = (0.0468)
                std = (0.108)
            else:
                mean = (0.0460, 0.0464, 0.0516)
                std = (0.1019, 0.1028, 0.1163)

            transform.append(T.Normalize(mean, std))

    # transform.append(T.RandomAdjustSharpness(sharpness_factor=2))
    # transform.append(T.RandomAutocontrast(p=0.5))

    transform = T.Compose(transform)

    dataset = ImageFolder(root=root, transform=transform)

    # uncomment following lines to address standardzation (i.e. mean=0, std=1)
    # dataloader = DataLoader(
    #     dataset=dataset,
    #     batch_size=48,
    # )

    # get_normalization_data(loader=dataloader)

    return dataset


def generate_random_sample(dataset, batch_size: int):
    """Generate batch of random samples from dataset.

    @params dataset (torchvision.datasets.ImageFolder): 
        dataset to sample from
    @params batch_size (int):
        size of the batch
    """
    while True:
        random_indexes = np.random.choice(dataset.__len__(), size=batch_size, replace=False)
        # INFO: dataset[x][y] -> x is the index of the image, y is the element of the tuple (img, label) to take.
        batch = [dataset[i][0] for i in random_indexes]
        yield torch.stack(batch, dim=0)


def weights_init(m) -> None:
    """Initialize network weights randomly.

    @params m: the network

    @usage: net.apply(weights_init)
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def label_smoothing(labels: torch.Tensor) -> torch.Tensor:
    """Smooth the labels.

    @params labels: the tensor of labels to smooth

    @returns labels: the smoothed labels
    """
    labels = torch.stack([ l - 0.3 + torch.rand(1, device=labels.device) * 0.5 if l == 1 else l + torch.rand(1, device=labels.device) * 0.3 for l in labels ], dim=0)
    return labels


def instance_noise(labels: torch.Tensor, p_flip: float = 0.05) -> torch.Tensor:
    """Flip labels randomly.

    @params labels: the tensor of labels to flip
    @params p_flip: the probability of flipping the labels

    @returns labels: the flipped labels
    """
    num2flip = int(p_flip * int(labels.size(dim=0)))
    flip_idx = np.random.choice(a=[l for l in range(labels.size(dim=0))], size=num2flip, replace=False)
    labels = torch.as_tensor(
        [
            (1 - labels[i]) if i in flip_idx else labels[i]
            for i in range(labels.size(dim=0))
        ],
        device=labels.device,
    )
    return labels


def plot_grad_flow(model) -> None:
    """Plot the gradients flowing through different layers in the net during training. Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in training loop after loss.backwards() to visualize the gradient flow
    
    @params model: nn.Module
        The neural network model to plot the gradients of
    """
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in model.named_parameters():
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())                          
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02)
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], 
               ['max-gradient', 'mean-gradient', 'zero-gradient'],
                loc='best')
