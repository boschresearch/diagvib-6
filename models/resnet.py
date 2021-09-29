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

import torch
import torch.nn as nn

import torchvision.models as models
from diagvibsix.dataset.config import OBJECT_ATTRIBUTES


class ResNet(nn.Module):
    """A ResNet18-based model.
    """

    def __init__(self, shape, task, model='resnet18', pretrained=False,
                 fixed=False, features_factor=1):
        super(ResNet, self).__init__()
        self.model = model
        self.shape = shape
        self.in_ch = 3 * shape[0]
        self.task = task
        self.factor_classes = len(OBJECT_ATTRIBUTES[self.task])

        self.resnet = getattr(models, self.model)()
        if pretrained:
            self._load_state_dict()

        # Remove final fc layer of resnet and replace this by fully connected layer.
        self.class_head = nn.Linear(self.resnet.fc.in_features * features_factor,
                                    self.factor_classes)
        self.resnet.fc = nn.Identity()

        # If fixed=True, then we don't want to further train the feature extractor
        if fixed:
            self._freeze_module(self.resnet)

    def _freeze_module(self, module):
        for m in module.modules():
            for param in m.parameters():
                param.requires_grad = False

    def _load_state_dict(self,
                         savedir=None):
        # Search for this models statedict in savedir
        f = [file for file in os.listdir(savedir) if self.model in file][0]
        self.resnet.load_state_dict(torch.load(os.path.join(savedir, f)))

    def encoder(self, image):
        feat = self.resnet(image)
        return feat

    def classifier(self, feat):
        return self.class_head(feat)

    def forward(self, image):
        feat = self.encoder(image)
        return self.classifier(feat)
