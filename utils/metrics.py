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

import copy
import os

import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt
from diagvibsix.dataset.config import OBJECT_ATTRIBUTES

matplotlib.use('Agg')


__all__ = ['Metrics', 'Losses']


def confusion_matrix(true, pred, num_classes):
    mask = (true >= 0) & (true < num_classes)
    cm = torch.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    return cm


def cm(pred, true):
    n_classes = pred.shape[1]
    pred = torch.argmax(pred, dim=1)
    return confusion_matrix(true.flatten(), pred.flatten(), n_classes)


def mean_accuracy(pred, true):
    n_classes = pred.shape[1]
    pred = torch.argmax(pred, dim=1)
    cm = confusion_matrix(true.flatten(), pred.flatten(), n_classes)

    correct = torch.diag(cm).sum()
    total = cm.sum()
    acc = correct / (total + 1e-10)
    return acc.item()


def per_class_accuracy(pred, true):
    n_classes = pred.shape[1]
    pred = torch.argmax(pred, dim=1)
    cm = confusion_matrix(true.flatten(), pred.flatten(), n_classes)

    correct_per_class = torch.diag(cm)
    total_per_class = cm.sum(dim=1)
    per_class_acc = correct_per_class / total_per_class
    avg_per_class_acc = np.nanmean(per_class_acc)
    return avg_per_class_acc


class Losses(object):
    def __init__(self,
                 losses_to_log=['train', 'val'],
                 data_loader=None):
        self.data_loader = data_loader
        self.losses = {phase: [0.] for phase in losses_to_log}

    def push(self, loss, phase):
        self.losses[phase][-1] += loss

    def summarize(self, phase):
        self.losses[phase][-1] /= len(self.data_loader[phase])

    def log(self, save_path):
        self.plot_losses(save_path=save_path)

    def reset(self):
        for phase in self.losses:
            self.losses[phase].append(0.)

    def plot_losses(self, save_path):
        files = []
        plt.figure(figsize=(5, 3))
        for phase in self.losses.keys():
            y = self.losses[phase]
            x = list(range(len(y)))
            plt.plot(x, y, label=phase)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'Losses.pdf'))
        files.append(os.path.join(save_path, 'Losses.pdf'))
        plt.close('all')
        return files


class Metrics(object):
    def __init__(self, metrics_to_log, task, tags):
        self.metrics_to_log = metrics_to_log
        self.task = task
        self.tags = tags

        self.metrics = dict()

        # Get empty metrics dictionary. It  has format
        # self.metrics[phase][task][tag][metric][value/counter]
        self.reset(init=True)

    def reset(self, init=False):
        if init:
            self.metrics.clear()
            for phase, metrics in self.metrics_to_log.items():
                self.metrics[phase] = {}
                metrics_dict = {}
                for m in metrics:
                    metrics_dict[m] = {'value': [torch.zeros(len(OBJECT_ATTRIBUTES[self.task]),
                                                             len(OBJECT_ATTRIBUTES[self.task]))],
                                       'count': 0
                                       } if m == 'cm' else {'value': [0.], 'count': 0}
                self.metrics[phase] = {tag: copy.deepcopy(
                    metrics_dict) for tag in self.tags[phase]}
        else:
            for phase, metrics in self.metrics_to_log.items():
                for tag in self.tags[phase]:
                    for m in metrics:
                        m_init = torch.zeros(len(OBJECT_ATTRIBUTES[self.task]),
                                             len(OBJECT_ATTRIBUTES[self.task])) if m == 'cm' else 0.
                        self.metrics[phase][tag][m]['value'].append(m_init)
                        self.metrics[phase][tag][m]['count'] = 0.

    def push(self, pred, target, tags, phase):
        # Move network predictions and targets to cpu
        pred = pred.data.cpu()
        target = target.data.cpu()

        # Iterate over the tasks, tags and metrics. For each tag, we compute the metric only on those samples within a
        # minibatch, which have the desired tag. We then weight that metric with the number of samples. This way in
        # summarize() we  can simply divide by the 'count' (i.e. how often that tag actually occured) and get the mean.
        # For confusion matrices, values are summed up (not averaged), thus we don't need to weight it (weight=1)
        for tag in self.tags[phase]:
            tag_mask = np.where(np.array(tags) == tag, True, False)
            for metric in self.metrics_to_log[phase]:
                weight = 1. if metric == 'cm' else np.sum(tag_mask)
                if np.sum(tag_mask) > 0:
                    m = globals()[metric]
                    self.metrics[phase][tag][metric]['value'][-1] += m(
                        pred[tag_mask], target[tag_mask]) * weight
                    self.metrics[phase][tag][metric]['count'] += np.sum(tag_mask)

    def summarize(self, phase):
        for tag in self.tags[phase]:
            for metric in self.metrics_to_log[phase]:
                self.metrics[phase][tag][metric]['value'][-1] /= self.metrics[phase][tag][metric]['count']

    def log(self, phases, save_path):
        if not isinstance(phases, list):
            phases = [phases]

        # Log to files.
        for phase in phases:
            for tag in self.tags[phase]:
                for metric in self.metrics_to_log[phase]:
                    if metric == 'cm':
                        self.plot_cm(phase, tag,
                                     os.path.join(save_path, '{}_{}_{}_{}.pdf'.format(phase, self.task, tag, metric)))
                    else:
                        self.plot_metrics(phase, tag, metric,
                                          os.path.join(save_path, '{}_{}_{}_{}.pdf'.format(phase, self.task, tag, metric)))

    def plot_cm(self, phase, tag, save_path):
        classes = OBJECT_ATTRIBUTES[self.task]
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        plt.imshow(self.metrics[phase][tag]['cm']
                   ['value'][-1], cmap=plt.get_cmap('Blues'))
        ax.set_xticks([i for i in range(len(classes))])
        ax.set_yticks([i for i in range(len(classes))])
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(classes, rotation='horizontal', fontsize=7)
        plt.xlabel('Predicted label', fontsize=12)
        plt.ylabel('True label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close('all')

    def plot_metrics(self, phase, tag, metric, savepath):
        y = self.metrics[phase][tag][metric]['value']
        x = list(range(len(y)))
        plt.figure(figsize=(5, 3))
        plt.plot(x, y)
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title('{}_{}_{}'.format(phase, self.task, tag))
        plt.tight_layout()
        plt.savefig(savepath)
        plt.close('all')
