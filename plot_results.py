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
import numpy as np
import argparse
from diagvibsix.auxiliaries import load_obj, load_yaml, get_corr_pred
from diagvibsix.dataset.config import EXPERIMENT_SAMPLES, FACTORS
import matplotlib.pyplot as plt


def plot_cb(fig, heatmap):
    cb = fig.colorbar(heatmap)
    cb.set_label('OOD Accuracy', rotation=-90, va="bottom", fontsize=10)
    cb.outline.set_visible(False)


def save_plt(fig):
    savepath = os.path.join(RESULTS_PATH, 'plts')
    savename = '{}_{}.pdf'.format(ARGS.study.split('_')[-1], HP_ARGS['method'])
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    fig.savefig(os.path.join(savepath, savename), bbox_inches='tight', dpi=300)


def plot_matrix():
    # Gather data: 'ic' are the in-distribution test samples, 'violated' are the OOD test ssamples
    accuracies = {'ic': np.full((len(FACTORS), len(FACTORS)), np.nan),
                  'violated': np.full((len(FACTORS), len(FACTORS)), np.nan)}

    for experiment in os.listdir(os.path.join(RESULTS_PATH, ARGS.study)):
        exp_ics = []
        exp_violated = []
        corrs, preds = get_corr_pred(experiment)
        cue = list(set(corrs) - set(preds))[0]

        # Loop over experiment samples
        for dataset_sample in range(EXPERIMENT_SAMPLES):
            stats_path = os.path.join(RESULTS_PATH, ARGS.study, experiment, str(dataset_sample), HP_ARGS['method'],
                                      'stats.pkl')
            try:
                stats = load_obj(stats_path)
                exp_ics.append(stats['metrics']['test']['ic']['per_class_accuracy']['value'][-1])
                exp_violated.append(stats['metrics']['test']['violate ' + cue]['per_class_accuracy']['value'][-1])
            except:
                continue

        if exp_ics and exp_violated:
            accuracies['ic'][FACTORS.index(preds[0])][FACTORS.index(cue)] = np.mean(exp_ics)
            accuracies['violated'][FACTORS.index(preds[0])][FACTORS.index(cue)] = np.mean(exp_violated)

    # Plot the data in a matrix plot
    fig, ax = plt.subplots(figsize=(4.8, 4))
    hm = ax.imshow(accuracies['violated'], cmap=CMAP, vmin=0, vmax=1)
    ax.spines[:].set_visible(False)
    ax.set_xticks([i for i in range(len(FACTORS))])
    ax.set_yticks([i for i in range(len(FACTORS))])
    ax.set_xticklabels(FACTORS, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(FACTORS, rotation='horizontal', fontsize=10)
    ax.set_ylabel(r'Predicted factor $\mathcal{F}_{i}$', fontsize=14)
    ax.set_xlabel(r'Correlated factor $\mathcal{F}_{j}$', fontsize=14)
    ax.set_title('{}'.format(ARGS.study.split('_')[-1]), fontsize=16)

    # Add text annotations
    for i in range(len(FACTORS)):
        for j in range(len(FACTORS)):
            if i != j:
                annot_col = 'white' if accuracies['violated'][i, j] >= 0.5 else 'black'
                text = ax.text(j, i, "{:.2f}".format(accuracies['violated'][i, j]),
                               ha="center", va="center", color=annot_col)

    # Add colorbar and save plot
    plot_cb(fig, hm)
    save_plt(fig)


def plot_vector():
    accuracies = np.full((len(FACTORS),1), np.nan)
    stds = np.full((len(FACTORS),1), np.nan)
    for experiment in os.listdir(os.path.join(RESULTS_PATH, ARGS.study)):
        _, preds = get_corr_pred(experiment)
        accs = []
        for dataset_sample in range(EXPERIMENT_SAMPLES):
            stats_path = os.path.join(RESULTS_PATH, ARGS.study, experiment, str(dataset_sample), HP_ARGS['method'],
                                      'stats.pkl')
            try:
                stats = load_obj(stats_path)
            except:
                continue
            accs.append(stats['metrics']['test']['ic']['per_class_accuracy']['value'][-1])
        if accs:
            accuracies[FACTORS.index(preds[0])] = np.mean(accs)
            stds[FACTORS.index(preds[0])] = np.std(accs)

    fig, ax = plt.subplots(figsize=(2, 4))
    hm = ax.imshow(accuracies, cmap=CMAP, vmin=0, vmax=1)
    ax.spines[:].set_visible(False)
    ax.set_yticks([i for i in range(len(FACTORS))])
    ax.set_xticks([])
    ax.set_yticklabels(FACTORS, rotation='horizontal', fontsize=10)
    ax.set_ylabel(r'Predicted factor $\mathcal{F}_{i}$', fontsize=14)
    ax.set_title('{}'.format(ARGS.study.split('_')[-1]), fontsize=16)

    # Add text annotations
    for i in range(len(FACTORS)):
        annot_col = 'white' if accuracies[i] >= 0.5 else 'black'
        text = ax.text(0, i, "{:.2f} \n+-{:.2f}".format(accuracies[i][0], stds[i][0]),
                       ha="center", va="center", color=annot_col, fontsize=6)

    # Add colorbar and save plot
    plot_cb(fig, hm)
    save_plt(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default='study_ZSO',
                        help="study_name")
    parser.add_argument("--hp", default='trainer/resnet18config.yml',
                        help="yaml that stores the hyperparameters.")
    ARGS = parser.parse_args()

    HP_ARGS = load_yaml(ARGS.hp)
    RESULTS_PATH = HP_ARGS['results_path']
    CMAP = plt.get_cmap('RdPu').copy()
    CMAP.set_bad(color='lightgrey')

    if 'ZSO' in ARGS.study:
        plot_vector()
    else:
        plot_matrix()
