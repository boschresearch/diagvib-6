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

import argparse


def make_parser():
    parser = argparse.ArgumentParser()

    # -------------------------------   General Settings   ------------------------------#
    parser.add_argument('-dev', '--device', type=int, default=0,
                        help='Cuda device to use. If -1, cpu is used instead.')
    parser.add_argument('--results_path',
                        default='./tmp/results',
                        help='The general save path.')
    # -----------------------------------   Dataset   -----------------------------------#
    parser.add_argument('--study_folder',
                        default='./tmp/diagvibsix/studies',
                        help='Folder the study specification were generated in.')
    parser.add_argument('--study',
                        default='study_ZGO',
                        help='Name of the study.')
    parser.add_argument('--experiment',
                        default='CORR-hue-lightness_PRED-hue',
                        help='Name of the experiment.')
    parser.add_argument('--dataset_sample',
                        default='0',
                        help='The sample of the dataset.')
    parser.add_argument('--dataset_seed', type=int, default=1332,
                        help='Seed to use for dataset setup.')
    parser.add_argument('--cache', action='store_true',
                        help='Cache the dataset in a .pkl file. Only helpful if benchmarking several methods at once.')

    # -----------------------------------   Method   -----------------------------------#
    parser.add_argument('--method', default='ResNet18Trainer',
                        help='Method to use. This should be one of the classes in trainer/.')
    parser.add_argument('--class_criterion', default='ce',
                        help='Criterion used for classification.')

    # -----------------------------------   Training   -----------------------------------#
    parser.add_argument('--training_seed', type=int, default=1332,
                        help='Seed with which torch is initialized.')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Number of epochs to train for.')
    parser.add_argument('--mbs', type=int, default=128,
                        help='Minibatch size to use.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers to use for loading and pre-processing.')
    parser.add_argument('--optimizer', default='adam',
                        help='Optimizer to use. Must be one of sgd | adam | rmsprop')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    # Adam
    parser.add_argument('--adam_b1', type=float, default=0.9,
                        help='b1 to use for adam')
    parser.add_argument('--adam_b2', type=float, default=0.999,
                        help='b2 to use for adam')
    # SGD
    parser.add_argument('--sgd_momentum', type=float, default=0.,
                        help='Momentum factor for SGD training')
    parser.add_argument('--sgd_dampening', type=float, default=0.,
                        help='Dampening for momentum factor for SGD training')

    return parser
