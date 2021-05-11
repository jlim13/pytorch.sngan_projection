# Training script for tiny-imagenet.
# Again, this script has a lot of bugs everywhere.
import argparse
import datetime
import json
import os
import shutil

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm

import evaluation
import losses as L
from models.discriminators.snresnet64 import SNResNetConcatDiscriminator
from models.discriminators.snresnet64 import SNResNetProjectionDiscriminator
from models.generators.resnet64 import ResNetGenerator
from models import inception
import utils

from datasets import cifar10, mnist

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

# Copied from https://github.com/naoto0804/pytorch-AdaIN/blob/master/sampler.py#L5-L15
def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


# Copied from https://github.com/naoto0804/pytorch-AdaIN/blob/master/sampler.py#L18-L26
class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31




def decay_lr(opt, max_iter, start_iter, initial_lr):
    """Decay learning rate linearly till 0."""
    coeff = -initial_lr / (max_iter - start_iter)
    for pg in opt.param_groups:
        pg['lr'] += coeff


def get_args():
    parser = argparse.ArgumentParser()
    # Dataset configuration
    parser.add_argument('--cGAN', default=False, action='store_true',
                        help='to train cGAN, set this ``True``. default: False')
    parser.add_argument('--data_root', type=str, default='tiny-imagenet-200',
                        help='path to dataset root directory. default: tiny-imagenet-200')
    parser.add_argument('--batch_size', '-B', type=int, default=64,
                        help='mini-batch size of training data. default: 64')
    parser.add_argument('--eval_batch_size', '-eB', default=None,
                        help='mini-batch size of evaluation data. default: None')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for training data loader. default: 8')
    # Generator configuration
    parser.add_argument('--gen_num_features', '-gnf', type=int, default=64,
                        help='Number of features of generator (a.k.a. nplanes or ngf). default: 64')
    parser.add_argument('--gen_dim_z', '-gdz', type=int, default=128,
                        help='Dimension of generator input noise. default: 128')
    parser.add_argument('--gen_bottom_width', '-gbw', type=int, default=4,
                        help='Initial size of hidden variable of generator. default: 4')
    parser.add_argument('--gen_distribution', '-gd', type=str, default='normal',
                        help='Input noise distribution: normal (default) or uniform.')
    # Discriminator (Critic) configuration
    parser.add_argument('--dis_arch_concat', '-concat', default=False, action='store_true',
                        help='If use concat discriminator, set this true. default: False')
    parser.add_argument('--dis_emb', type=int, default=128,
                        help='Parameter for concat discriminator. default: 128')
    parser.add_argument('--dis_num_features', '-dnf', type=int, default=64,
                        help='Number of features of discriminator (a.k.a nplanes or ndf). default: 64')
    # Optimizer settings
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Initial learning rate of Adam. default: 0.0002')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='beta1 (betas[0]) value of Adam. default: 0.0')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='beta2 (betas[1]) value of Adam. default: 0.9')
    parser.add_argument('--lr_decay_start', '-lds', type=int, default=50000,
                        help='Start point of learning rate decay. default: 50000')
    # Training setting
    parser.add_argument('--seed', type=int, default=46,
                        help='Random seed. default: 46 (derived from Nogizaka46)')
    parser.add_argument('--max_iteration', '-N', type=int, default=100000,
                        help='Max iteration number of training. default: 100000')
    parser.add_argument('--n_dis', type=int, default=5,
                        help='Number of discriminator updater per generator updater. default: 5')
    parser.add_argument('--num_classes', '-nc', type=int, default=0,
                        help='Number of classes in training data. No need to set. default: 0')
    parser.add_argument('--loss_type', type=str, default='hinge',
                        help='loss function name. hinge (default) or dcgan.')
    parser.add_argument('--relativistic_loss', '-relloss', default=False, action='store_true',
                        help='Apply relativistic loss or not. default: False')
    parser.add_argument('--calc_FID', default=False, action='store_true',
                        help='If calculate FID score, set this ``True``. default: False')
    parser.add_argument('--transform_space', type=str, default=None,
                        help='Which space to smack a transformer')
    # Log and Save interval configuration
    parser.add_argument('--results_root', type=str, default='results',
                        help='Path to results directory. default: results')
    parser.add_argument('--no_tensorboard', action='store_true', default=False,
                        help='If you dislike tensorboard, set this ``False``. default: True')
    parser.add_argument('--no_image', action='store_true', default=False,
                        help='If you dislike saving images on tensorboard, set this ``True``. default: False')
    parser.add_argument('--checkpoint_interval', '-ci', type=int, default=1000,
                        help='Interval of saving checkpoints (model and optimizer). default: 1000')
    parser.add_argument('--log_interval', '-li', type=int, default=100,
                        help='Interval of showing losses. default: 100')
    parser.add_argument('--eval_interval', '-ei', type=int, default=1000,
                        help='Interval for evaluation (save images and FID calculation). default: 1000')
    parser.add_argument('--n_eval_batches', '-neb', type=int, default=100,
                        help='Number of mini-batches used in evaluation. default: 100')
    parser.add_argument('--n_fid_images', '-nfi', type=int, default=5000,
                        help='Number of images to calculate FID. default: 5000')
    parser.add_argument('--test', default=False, action='store_true',
                        help='If test this python program, set this ``True``. default: False')
    # Resume training
    parser.add_argument('--args_path', default=None, help='Checkpoint args json path. default: None')
    parser.add_argument('--gen_ckpt_path', '-gcp', default=None,
                        help='Generator and optimizer checkpoint path. default: None')
    parser.add_argument('--dis_ckpt_path', '-dcp', default=None,
                        help='Discriminator and optimizer checkpoint path. default: None')
    args = parser.parse_args()
    return args


def sample_from_data(args, device, data_loader):
    """Sample real images and labels from data_loader.

    Args:
        args (argparse object)
        device (torch.device)
        data_loader (DataLoader)

    Returns:
        real, y

    """

    real, y = next(data_loader)
    real, y = real.to(device), y.to(device)
    if not args.cGAN:
        y = None
    return real, y


def sample_from_gen(args, device, num_classes, gen):
    """Sample fake images and labels from generator.

    Args:
        args (argparse object)
        device (torch.device)
        num_classes (int): for pseudo_y
        gen (nn.Module)

    Returns:
        fake, pseudo_y, z

    """

    z = utils.sample_z(
        args.batch_size, args.gen_dim_z, device, args.gen_distribution
    )
    if args.cGAN:
        pseudo_y = utils.sample_pseudo_labels(
            num_classes, args.batch_size, device
        )
    else:
        pseudo_y = None

    fake = gen(z, pseudo_y)
    return fake, pseudo_y, z


def main():
    args = get_args()
    # CUDA setting
    if not torch.cuda.is_available():
        raise ValueError("Should buy GPU!")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True

    def _rescale(img):
        return img * 2.0 - 1.0

    def _noise_adder(img):
        return torch.empty_like(img, dtype=img.dtype).uniform_(0.0, 1/128.0) + img



    eval_dataset = cifar10.CIFAR10(root=args.data_root,
                                        train=False,
                                        download=True,
                                        transform=transforms.Compose([
                                            transforms.Resize(64),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ]),
                                        minority_classes = None,
                                        keep_ratio = None)
    eval_loader = iter(torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size,
                                                sampler=InfiniteSamplerWrapper(eval_dataset),
                                                 num_workers=args.num_workers,
                                                 pin_memory=True))


    print(' prepared datasets...')

    # Prepare directories.
    num_classes = len(eval_dataset.classes)
    args.num_classes = num_classes

    # initialize models.
    _n_cls = num_classes if args.cGAN else 0
    gen = ResNetGenerator(
        args.gen_num_features, args.gen_dim_z, args.gen_bottom_width,
        activation=F.relu, num_classes=_n_cls, distribution=args.gen_distribution
    ).to(device)
    if args.dis_arch_concat:
        dis = SNResNetConcatDiscriminator(args.dis_num_features, _n_cls, F.relu, args.dis_emb).to(device)
    else:
        dis = SNResNetProjectionDiscriminator(args.dis_num_features, _n_cls, F.relu, args.transform_space).to(device)
    inception_model = inception.InceptionV3().to(device) if args.calc_FID else None

    gen = torch.nn.DataParallel(gen)
    # dis = torch.nn.DataParallel(dis)

    opt_gen = optim.Adam(gen.parameters(), args.lr, (args.beta1, args.beta2))
    opt_dis = optim.Adam(dis.parameters(), args.lr, (args.beta1, args.beta2))

    # gen_criterion = getattr(L, 'gen_{}'.format(args.loss_type))
    # dis_criterion = getattr(L, 'dis_{}'.format(args.loss_type))
    gen_criterion = L.GenLoss(args.loss_type, args.relativistic_loss)
    dis_criterion = L.DisLoss(args.loss_type, args.relativistic_loss)

    print(' Initialized models...\n')

    if args.args_path is None:
        print ("Please specify weights to load")
        exit()
    else:
        print(' Load weights...\n')

        prev_args, gen, opt_gen, dis, opt_dis = utils.resume_from_args(
            args.args_path, args.gen_ckpt_path, args.dis_ckpt_path
        )
    args.n_fid_batches = args.n_eval_batches
    fid_score = evaluation.evaluate(
        args, 0, gen, device, inception_model, eval_loader, to_save=False
    )
    print (fid_score)



if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
