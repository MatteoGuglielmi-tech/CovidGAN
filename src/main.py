"""Wrapper for running the model."""
# WARNING: if linter gives errors on generator's further calls ignore them.
import datetime
import os
import re
import time

import torch
import torch.nn as nn
from loop import TrainLoop
from Models.Discriminator import Discriminator
from Models.EnchancedDiscriminator import RaLSGANDiscriminator
from Models.EnchancedGenerator import RaLSGANGenerator
from Models.Generator import Generator
from utilities import torchutils
from utilities import utils
from utilities.parse import opts


def main(opts):
    """Run the model.

    @param opts (argparse.Namespace):
        Command-line arguments.
    """
    logger = utils.logger_setup()
    
    torchutils.set_seed()
    score = re.findall(r"\d{1}(?=-root)", opts.dataset)[0]

    logger.info("Create and set-up models...")
    if not opts.improved:
        generator = Generator(opts)
        discriminator = Discriminator(opts)
    else:
        generator = RaLSGANGenerator(channels=opts.channels, hidden_dim=opts.hidden_dim)
        discriminator = RaLSGANDiscriminator(channels=opts.channels, hidden_dim=opts.hidden_dim)

    if torch.cuda.is_available():
        # for optimization purposes
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        cudnn.deterministic = True

        if opts.n_gpus > 1:
            generator = nn.DataParallel(discriminator, list(range(opts.n_gpus)))
            discriminator = nn.DataParallel(generator, list(range(opts.n_gpus)))

        generator = generator.cuda()
        discriminator = discriminator.cuda()

    logger.info("Loading the dataset...")
    dataset = torchutils.parse_data(
            root=opts.dataset,
            img_size=opts.img_size, 
            )

    if opts.training:
        writer = torchutils.writer_setup()

        train_exp_folder = utils.create_experiment_folder(opts=opts)
        print(f"\nExperiment traning folder: {train_exp_folder}\n")

        generator.apply(torchutils.weights_init)
        discriminator.apply(torchutils.weights_init)

        logger.info("Setting up optimizers and start training...")
        train_loop = TrainLoop(
                generator=generator,
                discriminator=discriminator,
                dataset=dataset,
                writer=writer,
                dest_folder=train_exp_folder
                )

        if opts.verbose:
            generator._verbose()
            discriminator._verbose()
            print(f"Optimizer G: {train_loop.optimizer_G}")
            print(f"Optimizer D: {train_loop.optimizer_D}")
            for arg in vars(opts):
                print(f"{arg}: {getattr(opts, arg)}")

        start_time = time.time()

        train_loop.train()
        generator.save_weights(path=f"weights/generator_weights_score-{score}.pth")

        end_time = time.time()
        logger.info(f"Training took {datetime.timedelta(seconds=end_time-start_time)}")

        logger.info("Processing training results ...")
        os.system("tensorboard --logdir=runs")
        utils.make_gif(folder=f"{train_exp_folder}", name="generator_progress", tot_iters=opts.iters)

    else:
        generator.load_weights(path=f"weights/generator_weights_score-{score}.pth")
        logger.info("Generating new samples ...")
        eval_exp_folder = utils.create_experiment_folder(opts=opts, mode="Evaluation")
        print(f"\nExperiment evaluation folder: {eval_exp_folder}\n")
        generator.evaluate(samples2generate=opts.samples2generate, expFolder=eval_exp_folder)

        logger.info("Computing similarity score ...")
        start_time = time.time()
        utils.compute_similarity_score(
                orig_folder=eval_exp_folder, 
                orig_dataset=eval_exp_folder, 
                gen_folder=eval_exp_folder, 
                filename=score, 
                score_names=opts.similarity_scores
                )
        end_time = time.time()
        logger.info(f"Training took {datetime.timedelta(seconds=end_time-start_time)}")
        os.system(f"python ./results/Evaluation/table_creator.py -ll ./results/Evaluation/{score}/{score}.log")


if __name__ == "__main__":
    main(opts)
    os.system("rm terminal.log")
