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

import numpy as np
import torch
import sys
from argparser import make_parser
import trainer
import random


def main():
    # Setup dataset seed.
    if args.dataset_seed is None:
        args.dataset_seed = np.random.randint(1, 10000)
    np.random.seed(args.dataset_seed)
    random.seed(args.dataset_seed)
    # Setup training seed.
    if args.training_seed is None:
        args.training_seed = np.random.randint(1, 10000)
    torch.manual_seed(args.training_seed)
    torch.cuda.manual_seed(args.training_seed)
    torch.backends.cudnn.deterministic = True

    # Setup trainer and run.
    method_trainer = getattr(trainer, args.method)
    this_trainer = method_trainer(args)
    this_trainer.run()


if __name__ == '__main__':
    # Argument parser.
    parser = make_parser()
    args = parser.parse_args()
    args.start_command = ' '.join(sys.argv)

    #Execute main
    main()
