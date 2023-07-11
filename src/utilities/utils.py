"""Utility file containing useful functions."""
import argparse
import logging
import math
import os
import re
from typing import List

import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from sewar.full_ref import msssim
from sewar.full_ref import psnr
from sewar.full_ref import psnrb
from sewar.full_ref import rase
from sewar.full_ref import rmse
from sewar.full_ref import scc
from tqdm import tqdm


def logger_setup(log_name: str="terminal") -> logging.Logger:
    """Return a logger object.

    @returns logger (logging.Logger): 
        logger object
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s -> %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_name + ".log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def create_experiment_folder(opts: argparse.Namespace, mode: str="Training", additional: str="") -> str:
    """Create a folder for the experiment.

    @params opts (argparse.Namespace): 
        options form CLI
    @params mode (str, optional):
        mode of the experiment (choices are "Training" or "Evaluation")
    @params additonal(str, optional):
        additional information to append to folder name

    @returns folder_name (str):
        name of the folder
    """
    assert mode in ["Training", "Evaluation"], "Invalid mode. Choose between 'Training' or 'Evaluation'."

    score = re.search(r"\d+(?=-root)", opts.dataset).group()
    folder_name = f"{os.getcwd()}/results/{mode}/{score}"
    try:
        os.mkdir(folder_name)
    except FileExistsError:
        pass
    
    folder_name += f"/{opts.model}"
    if opts.improved:
        folder_name += f"-improved-b{opts.batch_size}-c{opts.channels}-latent{opts.latent_dim}-i{opts.iters}-lrG{opts.lrG}-lrD{opts.lrD}-dcG{opts.decayG}-dcD{opts.decayD}-hd{opts.hidden_dim}"
    else: 
        folder_name += f"-b{opts.batch_size}-c{opts.channels}-latent{opts.latent_dim}-i{opts.iters}-lrG{opts.lrG}-lrD{opts.lrD}-dcG{opts.decayG}-dcD{opts.decayD}"
    
    if opts.noisy_discr_input:
        folder_name += "-noisy_discr_input"
    if opts.label_smoothing:
        folder_name += "-label_smoothing"
    if opts.instance_noise:
        folder_name += "-instance_noise"
    if len(additional) > 0:
        folder_name += f"-{additional}"

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        
    return folder_name


def make_gif(folder: str, name: str, duration: float=500, tot_iters: int=4000) -> None:
    """Make a gif from the images in the folder.

    @params folder (str): 
        folder containing images
    @params name (str):
        name of the gif
    @params duration (float, optional):
        duration of each frame in the gif in ms
    """
    images = []
    font = ImageFont.load_default()
    with os.scandir(folder) as it:
        for idx, entry in enumerate(it):
            if entry.name.endswith(".png") and entry.is_file():
                img = Image.open(entry.path)
                draw = ImageDraw.Draw(img)

                draw.text((0, 0), f"Iteration [{str(idx*50)}/{tot_iters}]", (255, 255, 255), font=font)
                images.append(img)

    images[0].save(
        f"{folder}/{name}.gif",
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )


def compute_similarity_score(orig_folder: str, orig_dataset, gen_folder: str, filename: str, score_names: List[str]=["msssim"], maxBool: bool=True) -> None:
    """Compute similarity score between original with synthetic images.

    @params orig_folder (ImageFolder dataset): 
        folder containing original images
    @params gen_folder (str): 
        folder containing generated images
    @params score_name (List[str], optional):
        name of the similarity scores to compute
    @params maxBool (bool, optional):
        whether the maximum similarity score corresponds to perfect match or not

    @returns db (Dict[str, float]):
        dictionary containing the similarity score for each image where key is the image name and value is the similarity score.
    """
    local_logger = logger_setup(log_name=filename)
    db = { score_name: {} for score_name in score_names }
    orig_names = [item.name for item in os.scandir(orig_folder) if item.name.endswith(".png") and item.is_file()]
    fake_names = [item.name for item in os.scandir(gen_folder) if item.name.endswith(".png") and item.is_file()]

    with tqdm(total=len(orig_names), unit="image") as pbar:
        for idx in range(len(orig_names)):
            pbar.update(1)
            pbar.set_description(f"[ %d / %d]" % (idx + 1, len(orig_names)))
            orig_name = orig_names[idx]
            if isinstance(orig_dataset, str):
                orig_img = np.asarray(Image.open(os.path.join(orig_folder, orig_name)).convert('L')).reshape((256, 256, 1)) 
            else:
                orig_img = (255 * torch.reshape(orig_dataset[idx][0], (256, 256, 1)).numpy()).astype(np.uint8)
            for id in range(len(fake_names)):
                fake_name = fake_names[id]
                fake_img = np.asarray(Image.open(os.path.join(gen_folder, fake_name)).convert('L')).reshape((256, 256, 1))

                for score_name in score_names:
                    if score_name == "msssim":
                        score = msssim(orig_img, fake_img)
                        # compute module of complex number 
                        score = math.sqrt(score.real ** 2 + score.imag ** 2) 
                        db[score_name][(orig_name, fake_name)] = score
                    elif score_name == "rmse":
                        score = rmse(orig_img, fake_img)
                        db[score_name][(orig_name, fake_name)] = score
                    elif score_name == "sam":
                        score = scc(orig_img, fake_img)
                        db[score_name][(orig_name, fake_name)] = score
                    elif score_name == "psnr":
                        score = psnr(orig_img, fake_img)
                        db[score_name][(orig_name, fake_name)] = score
                    elif score_name == "psnrb":
                        score = psnrb(orig_img, fake_img)
                        db[score_name][(orig_name, fake_name)] = score
                    elif score_name == "rase":
                        score = rase(orig_img, fake_img)
                        db[score_name][(orig_name, fake_name)] = score
                    else:
                        raise NotImplementedError(f"Similarity score {score_name} not implemented.")

            pbar.set_postfix_str(f"{orig_name}")

    res = {}
    for score_name in score_names:
        if maxBool:
            most_similar = max(db.get(score_name, {}).values())
        else: 
            most_similar = min(db.get(score_name, {}).values())

        for (org_key, fake_key), value in db.get(score_name, {}).items():
            local_logger.info(f"{score_name} score for {fake_key} with {org_key} is : {value}")
            if value == most_similar:
                res[(org_key, fake_key)] = value
        if not isinstance(orig_dataset, str):
            for (org_key, fake_key), value in res.items():
                local_logger.info(f"Most similar image is {fake_key} with {org_key} with a {score_name} score of {value}")
