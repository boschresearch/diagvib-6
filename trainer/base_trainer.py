#!/usr/local/bin/python3
# Copyright (c) 2021 Robert Bosch GmbH Copyright holder of the paper "DiagViB-6: A Diagnostic Benchmark Suite for Vision Models in the Presence of Shortcut and Generalization Opportunities" accepted at ICCV 2021.
# All rights reserved.
###
# The paper "DiagViB-6: A Diagnostic Benchmark Suite for Vision Models in the Presence of Shortcut and Generalization Opportunities" accepted at ICCV 2021.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Author: Elias Eulig, Volker Fischer
# -*- coding: utf-8 -*-

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from diagvibsix import TorchDatasetWrapper
from diagvibsix.auxiliaries import save_obj, save_yaml
from utils.metrics import Losses, Metrics

__all__ = [
    'setup_optimizer',
    'setup_loss',
    'setup_dataloader',
    'BaseTrainer',
]


def setup_optimizer(args, parameters):
    """Setup the optimizer.

    Args:
        args (argparser.Namespace): Namespace containing optimizer settings.

    Returns:
        torch.nn.Module: Class instance for the respective optimizer.

    """

    if args.optimizer == 'sgd':
        return optim.SGD(parameters,
                         lr=args.lr,
                         momentum=args.sgd_momentum,
                         dampening=args.sgd_dampening
                         )
    elif args.optimizer == 'adam':
        return optim.Adam(parameters,
                          lr=args.lr,
                          betas=(args.adam_b1, args.adam_b2))
    elif args.optimizer == 'rmsprop':
        return optim.RMSprop(parameters,
                             lr=args.lr)
    else:
        raise ValueError('Optimizer unknown. Must be sgd | adam | rmsprop')


def setup_loss(loss, **kwargs):
    """Setup the loss.

    Args:
        loss (str): Loss function to use

    Returns:
        torch.nn.Module: Class instance for the respective loss.

    """

    if loss == 'ce':
        return nn.CrossEntropyLoss()
    elif loss == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss == 'mse':
        return nn.MSELoss(reduction='mean')
    else:
        raise ValueError('Loss function unknown.')


def setup_dataloader(args, datasets):
    """Given the main args and a dict of datasets returns respective dataloaders.

    Args:
        loss (str): Loss function to use

    Returns:
        torch.nn.Module: Class instance for the respective loss.

    """

    data_loader = {phase: DataLoader(dataset=data, batch_size=args.mbs, num_workers=args.num_workers,
                                     pin_memory=args.device >= 0, shuffle=True, drop_last=True)
                   for phase, data in datasets.items()}
    return data_loader


class BaseTrainer(object):
    """Base class to benchmark a model.

    This is the base class to benchmark a model on DiagViB-6. All model-specific
    trainers should be inheretid from this base class and then overwrite methods
    if needed (see trainer/trainer_setup.py for an example).

    Attributes:
        args (argparser.Namespace): Namespace containing training arguments
        dev (torch.device): Device to perform training on
        benchmark_path (string): Benchmark path for this study/experiment/sample
        results_path (string): Path to store results to
        specs (dict): Dict storing the paths to dataset specifications
        data (dict): Dict storing the datasets
        data_loader (dict): Dict storing the data loaders
        class_criterion (torch.nn.module): Torch loss module
        tags (dict): Dict containing the tags for each phase
        task (str): String of the task (e.g. shape)
        shape (tuple): Tuple of input image shape
        losses (Losses): Losses object tracking training and validation losses
        metrics (Metrics): Metrics object tracking train/val/test metrics
        model (None): Placeholder for the actual model
        optimizer (None): Placeholder for the actual optimizer

    """

    def __init__(self, args):
        self.args = args
        self.dev = torch.device(
            'cuda', args.device) if args.device >= 0 else torch.device('cpu')

        self.benchmark_path = os.path.join(args.study, args.experiment,
                                           args.dataset_sample)
        self.results_path = os.path.join(self.args.results_path,
                                         self.benchmark_path, self.args.method)

        # Generate paths to dataset specifications.
        study_path = os.path.join(args.study_folder, self.benchmark_path)
        self.specs = {t: os.path.join(study_path, t + '.yml')
                      for t in ['train', 'val', 'test']}

        self.data = {'train': TorchDatasetWrapper(self.specs['train'],
                                                  args.dataset_seed,
                                                  cache=args.cache)}
        self.data = {**self.data, **{phase: TorchDatasetWrapper(self.specs[phase], args.dataset_seed + i + 1,
                                                                mean=self.data['train'].mean,
                                                                std=self.data['train'].std,
                                                                cache=args.cache)
                                     for i, phase in enumerate(['val', 'test'])}}

        # Setup data loader.
        self.data_loader = setup_dataloader(args, self.data)

        # Setup criterion.
        self.class_criterion = setup_loss(args.class_criterion)

        # Get train, val and test tags and task
        self.tags = {
            phase: self.data[phase].tags for phase in self.specs.keys()}
        self.task = self.data['train'].task

        # Get shape
        self.shape = self.data['train'].shape

        # Setup logging information
        self.losses = Losses(losses_to_log=['train', 'val'],
                             data_loader=self.data_loader)

        metrics_to_log = {'train': ['per_class_accuracy', 'cm'],
                          'val': ['per_class_accuracy', 'cm'],
                          'test': ['per_class_accuracy']}
        self.metrics = Metrics(metrics_to_log, self.task, self.tags)

        # Setup placeholders for model and optimizer
        self.model = None
        self.optimizer = None

        self.epoch = 0

    def train_step(self, batch, tags):
        input, target = batch

        self.optimizer.zero_grad()
        out = self.model(input)

        loss = self.class_criterion(out, target)

        loss.backward()
        self.optimizer.step()

        self.losses.push(loss.item(), 'train')
        self.metrics.push(out, target, tags, 'train')

    def train(self):
        self.model.train()
        for i_batch, batch in enumerate(tqdm(self.data_loader['train'], desc='Train')):
            tags = batch.pop('tag')
            batch = (Variable(v).to(self.dev, non_blocking=True)
                     for v in batch.values())
            self.train_step(batch, tags)

        self.losses.summarize('train')
        self.metrics.summarize('train')

    def val_step(self, i_batch, batch, tags):
        input, target = batch

        out = self.model(input)

        loss = self.class_criterion(out, target)

        self.losses.push(loss.item(), 'val')
        self.metrics.push(out, target, tags, 'val')

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for i_batch, batch in enumerate(tqdm(self.data_loader['val'], desc='Validate')):
                tags = batch.pop('tag')
                batch = (v.to(self.dev, non_blocking=True)
                         for v in batch.values())
                self.val_step(i_batch, batch, tags)

        self.losses.summarize('val')
        self.metrics.summarize('val')

    def test_step(self, batch, tags):
        input, target = batch
        out = self.model(input)
        self.metrics.push(out, target, tags, 'test')

    def test(self):
        self.model.eval()
        with torch.no_grad():
            for i_batch, batch in enumerate(tqdm(self.data_loader['test'], desc='Test')):
                tags = batch.pop('tag')
                batch = (v.to(self.dev, non_blocking=True)
                         for v in batch.values())
                self.test_step(batch, tags)

        self.metrics.summarize('test')

    def save_checkpoint(self):
        save_path = os.path.join(self.results_path, 'checkpoints')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for phase in ['val']:
            if self.epoch == np.argmin(self.losses.losses[phase]):
                checkpoint_path = os.path.join(
                    save_path, 'best_{}.pt'.format(phase))
                torch.save({'epoch': self.epoch,
                            'model_state_dict': self.model.state_dict()
                            }, checkpoint_path
                           )

    def load_checkpoint(self, path, only_weights=False):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if not only_weights:
            self.epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def log_stats(self):
        # Log losses and metrics
        plts_save_path = os.path.join(self.results_path, 'plts')
        if not os.path.exists(plts_save_path):
            os.makedirs(plts_save_path)
        self.metrics.log(phases=['train', 'val'], save_path=plts_save_path)
        self.losses.log(save_path=plts_save_path)

    def fit(self):
        self.start_time = time.time()

        for epoch in trange(self.epoch, self.args.num_epochs):
            torch.manual_seed(self.args.training_seed + epoch)
            # Train val test loop
            self.train()
            self.validate()

            # Logging
            self.log_stats()
            self.save_checkpoint()

            # Losses and metrics reset
            self.metrics.reset()
            self.losses.reset()
            self.epoch += 1

        time_elapsed = time.time() - self.start_time
        print('Training completed in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    def run(self):
        # Create results directory and save args + start command
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
        save_yaml(vars(self.args), os.path.join(self.results_path, 'args.yaml'))

        # Run training.
        self.fit()

        # Load best validation net and run test
        print('Run test using best validation network ...')
        checkpoint_path = os.path.join(self.results_path, 'checkpoints',
                                       'best_val.pt')
        self.load_checkpoint(checkpoint_path, only_weights=True)
        self.test()
        self.metrics.log(phases='test',
                         save_path=os.path.join(self.results_path, 'plts'))

        # Save losses and metric histories to pkl file
        save_obj({'metrics': self.metrics.metrics, 'losses': self.losses.losses},
                 os.path.join(self.results_path, 'stats.pkl'))
