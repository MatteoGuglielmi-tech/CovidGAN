"""Parser for the command line arguments."""
import argparse


dataroot0 = "../Dati-San-Matteo-2/0-root"
dataroot1 = "../Dati-San-Matteo-2/1-root"
dataroot2 = "../Dati-San-Matteo-2/2-root"
dataroot3 = "../Dati-San-Matteo-2/3-root"

parser = argparse.ArgumentParser(
        prog="GAN",
        usage=None,
        description='Generative Adversarial Networks',
        add_help=True,
        allow_abbrev=True,
        )

parser.add_argument(
        '--model',
        '-m',
        type=str,
        choices=['RSGAN', 'RaSGAN', 'RaLSGAN', 'RaHingeGAN'],
        required=True,
        default='RaLSGAN',
        help='Model to use for training',
        )

parser.add_argument(
        '--improved',
        '-imp',
        action='store_true',
        help='Whether to use improved version of the model with SELU and spectral norm.',
        )

parser.add_argument(
        '--no_improved',
        '-nimp',
        dest='improved',
        action='store_false',
        )

parser.set_defaults(improved=True)

parser.add_argument(
        '--dataset', 
        '-d',
        type=str,
        choices=[dataroot0, dataroot1, dataroot2, dataroot3],
        required=True,
        help='Path to the dataset folder.',
        )

parser.add_argument(
        '--training',
        '-t',
        action='store_true',
        help='Whether to train the model.',
        )

parser.add_argument(
        '--evaluation',
        '-e',
        dest='training',
        action='store_false',
        help='Whether to evaluate the model.',
        )

parser.set_defaults(training=True)

parser.add_argument(
        '--batch_size',
        '-b',
        type=int,
        default=32,
        required=False,
        help='Size of the batches',
        )

parser.add_argument(
        '--channels',
        '-c',
        type=int,
        default=3,
        required=False,
        help='Number of image channels',
        )

parser.add_argument(
        '--noisy_discr_input',
        '-ndi',
        action='store_true',
        help='Whether to add noise to the discriminator input.',
        )

parser.add_argument(
        '--no_noise_discr_input',
        '-nndi',
        dest='noisy_discr_input',
        action='store_false',
        )

parser.set_defaults(noisy_discr_input=False)

parser.add_argument(
        '--label_smoothing',
        '-lb',
        action='store_true',
        help='Whether to smooth labels while training discr.',
        )

parser.add_argument(
        '--no_smoothing',
        '-nlb',
        dest='label_smoothing',
        action='store_false',
        )

parser.set_defaults(label_smoothing=True)

parser.add_argument(
        '--instance_noise',
        '-in',
        action='store_true',
        help='Whether to add instance noise while training discr.',
        )

parser.add_argument(
        '--no_instance_noise',
        '-nin',
        dest='instance_noise',
        action='store_false',
        )

parser.set_defaults(instance_noise=True)

parser.add_argument(
        '--iters',
        '-i',
        type=int,
        default=20e3,
        required=False,
        help='Number of training iterations',
        )

parser.add_argument(
        '--itersD',
        '-iD',
        type=int,
        default=1,
        required=False,
        help='Number of discriminator iterations',
        )

parser.add_argument(
        '--itersG',
        '-iG',
        type=int,
        default=1,
        required=False,
        help='Number of generator iterations',
        )

parser.add_argument(
        '--lrG',
        '-lrG',
        type=float,
        default=0.0002,
        required=False,
        help='[Adam] : learning rate for generator',
        )

parser.add_argument(
        '--lrD',
        '-lrD',
        type=float,
        default=0.0002,
        required=False,
        help='[Adam] : learning rate for discriminator',
        )

parser.add_argument(
        '--beta1',
        '-b1',
        type=float,
        default=0.5,
        required=False,
        help='[Adam] : decay rate for first moment estimate',
        )

parser.add_argument(
        '--beta2',
        '-b2',
        type=float,
        default=0.999,
        required=False,
        help='[Adam] : decay rate for second moment estimate',
        )

parser.add_argument(
        '--decayG',
        '-dcG',
        type=float,
        default=0,
        help='Decay to apply to lrG each cycle.'
        )

parser.add_argument(
        '--decayD',
        '-dcD',
        type=float,
        default=0,
        help='Decay to apply to lrD each cycle.'
        )

parser.add_argument(
        '--latent_dim',
        '-ld',
        type=int,
        default=128,
        required=False,
        help='Dimensionality of the latent space',
        )

parser.add_argument(
        '--hidden_dim',
        '-hd',
        type=int,
        default=32,
        required=False,
        help='Defined hidden layers of improved discrminator and generator'
        )

parser.add_argument(
        '--img_size',
        '-is',
        type=int,
        default=256,
        required=False,
        help='Size of output image',
        )

parser.add_argument(
        '--normalize',
        '-n',
        action='store_true',
        required=False,
        help="Whether apply tranform.Normalize() to the dataset or not."
        )

parser.add_argument(
        "--no-normalize",
        "-nn",
        dest="normalize",
        action="store_false",
        )

parser.set_defaults(normalize=True)

parser.add_argument(
        '--standard_normalization',
        '-sn',
        action='store_true',
        help="Whether to use standard or tailored on normalization."
        )

parser.add_argument(
        "--no-standard_normalization",
        "-nsn",
        dest="standard_normalization",
        action="store_false",
        )

parser.set_defaults(standard_normalization=True)

parser.add_argument(
        '--full_scale',
        '-fs',
        action='store_true',
        required=False,
        help='Whether to use full scale images or not.',
        )

parser.add_argument(
        "--no-full_scale",
        "-nfs",
        dest="full_scale",
        action="store_false",
        )

parser.set_defaults(full_scale=False)

parser.add_argument(
        '--n_gpus',
        '-ng',
        type=int,
        default=1,
        required=False,
        help='Number of GPUs available.',
        )

parser.add_argument(
        '--sample_interval',
        '-si',
        type=int,
        default=100,
        required=False,
        help='Interval between image sampling',
        )

parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        required=False,
        help="Whether to show hyperparameters or not."
        )

parser.add_argument(
        "--no-verbose",
        "-nv",
        dest="verbose",
        action="store_false",
        )

parser.set_defaults(verbose=False)

parser.add_argument(
        '--random_seed',
        '-rs',
        action='store_true',
        help="Whether to use a random seed or not."
        )

parser.add_argument(
        "--no-random_seed",
        "-nrs",
        dest="random_seed",
        action="store_false",
        )

parser.set_defaults(random_seed=False)

parser.add_argument(
        '--generate_progress-gif',
        '-gpg',
        action='store_true',
        help="Whether to generate a gif with the generator progress of the training."
        )

parser.add_argument(
        "--no-generate_progress-gif",
        "-ngpg",
        dest="generate_progress_gif",
        action="store_false",
        )

parser.set_defaults(generate_progress_gif=True)

opts = parser.parse_args()

# img_shape = (opts.channels, opts.img_size, opts.img_size)
# n_features = opts.channels * opts.img_size * opts.img_size
