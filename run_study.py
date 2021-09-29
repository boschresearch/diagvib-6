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
import os

from diagvibsix.auxiliaries import load_yaml
from diagvibsix.dataset.config import EXPERIMENT_SAMPLES, SHARED_STUDY_PATH


def load_hyperparameters(savepath):
    """This loads the hyperparameter yaml and converts it to a string of
    argparser arguments."""
    # Load yaml
    hp_args = load_yaml(savepath)

    # Generate argument string
    argstr = ''
    for i, (key, value) in enumerate(hp_args.items()):
        argstr += '--{} {}'.format(key, value)
        if i < len(hp_args) - 1:
            argstr += ' '
    return argstr


def run_study(hp_file, study_name, exp=None, sample=None):
    argstring = load_hyperparameters(hp_file)
    for experiment in os.listdir(os.path.join(SHARED_STUDY_PATH, study_name)):
        if exp and experiment != exp:
            continue

        for dataset_sample in range(EXPERIMENT_SAMPLES):
            if sample and str(dataset_sample) != sample:
                continue

            print('Train {} | Experiment {} | Sample {}'.format(
                study_name, experiment, dataset_sample))
            # YOU MAY WANT TO CHANGE THIS TO A SUBMIT CALL TO RUN DiagViB-6 ON A CLUSTER
            cmd = "python run.py {args} --dataset_seed {seed} --study_folder {study_folder} --study {study} --experiment {exp} --dataset_sample {samp}".format(
                args=argstring,
                seed=1332 + dataset_sample * 10,
                study_folder=SHARED_STUDY_PATH,
                study=study_name, exp=experiment,
                samp=dataset_sample
                )
            os.system(cmd)


def main():
    """This scripts runs a baseline on a single study or on experiments of a
    study. Hyperparameters should be provided through yaml files.

    Examples:
        - To run a new baseline on the ZGO study, create a .yml file for your
        baseline where you store all arguments from argparser.py that are not
        dataset related. Then run e.g.
        python studies/run_studies.py --study study_ZGO --hp trainer/FancyArchitecture.yml
        - To run a baseline only on the task where hue and position are
        correlated and the position is predicted in the ZGO study, run:
        python studies/run_studies.py --study study_ZGO/CORR-position-hue_PRED-position --hp trainer/FancyArchitecture.yml

    Depending on the study and your model, the benchmark may take very long if
    only run on a single machine. We recommend running DiagViB-6 on a cluster.
    For this, you'll want to change l. 36 to a call to a submit script for your
    cluster.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default='study_ZSO',
                        help="<study_name>/<experiment_name>/<dataset_sample>")
    parser.add_argument("--hp", default='trainer/resnet18config.yml',
                        help="yaml that stores the hyperparameters. We store those for each baseline in ../args_0-1")
    args = parser.parse_args()

    study_split = args.study.split('/')
    run_study(args.hp, *study_split)


if __name__ == '__main__':
    main()
