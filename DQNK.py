# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:29:37 2024

@author: Mamiya
这个脚本用来实现一个把所有MLP层都换成KAN层的DQN
"""
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from xuance.environment import DummyVecEnv
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.learners import DQN_Learner
from xuance.torch.agents import Agent
from xuance.common import DummyOffPolicyBuffer, DummyOffPolicyBuffer_Atari
from xuance.torch.agents.qlearning_family.dqn_agent import DQN_Agent
from torch import nn
from xuance.common import get_time_string, create_directory, RunningMeanStd, space2shape, EPS
from K import KAN,KBasicQnetwork,Basic_KAN
ActivationFunctions["silu"] = torch.nn.SiLU
REGISTRY_Policy["KBasic_Q_network"] = KBasicQnetwork
class KDQN_Agent(DQN_Agent):
    def _Kbuild_representation(self, representation_key: str, config: Namespace):
        normalize_fn = NormalizeFunctions[config.normalize] if hasattr(config, "normalize") else None
        initializer = nn.init.orthogonal_
        activation = ActivationFunctions[config.activation]
        input_shape = space2shape(self.observation_space)
        if representation_key == "KAN":
            grid_size = 5;spline_order = 3;
            if hasattr(config, "grid_size"):
                grid_size = config.grid_size
            if hasattr(config,"spline_order"):
                spline_order = config.spline_order
            self.representation_hidden_size = [input_shape[0]] + self.config.representation_hidden_size
            representation = KAN(layers_hidden = self.representation_hidden_size,grid_size = grid_size,spline_order = spline_order,device = self.device )
            representation.output_shapes = {'state':(representation.layers_hidden[-1],)}
            return representation
        else:
            raise AttributeError("Use KAN as representation")
    def _build_policy(self):
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device

        # build representation.
        if self.config.policy == "Basic_Q_network":
            configcopy = deepcopy(self.config)
            configcopy.representation = "Basic_MLP"
            representation = self._build_representation(configcopy.representation, configcopy)
            policy = REGISTRY_Policy["Basic_Q_network"](
                action_space=self.action_space, representation=representation, hidden_size=self.config.q_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device)
            self.config.policy = "KBasic_Q_network"
            #return policy
        # build policy.
        elif self.config.policy == "KBasic_Q_network":
            representation = self._Kbuild_representation(self.config.representation, self.config)
            grid_size = 5;spline_order = 3;
            if hasattr(self.config, "grid_size"):
                grid_size = self.config.grid_size
            if hasattr(self.config,"spline_order"):
                spline_order = self.config.spline_order
            Size = {"grid_size":grid_size,"spline_order":spline_order}
            policy = REGISTRY_Policy["KBasic_Q_network"](
                action_space=self.action_space, representation=representation, hidden_size=self.config.q_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device,size = Size)
        else:
            raise AttributeError(f"{self.config.agent} currently does not support the policy named {self.config.policy}.")

        return policy
    
#下面是测试
# import argparse
# from xuance.common import get_configs
# from xuance.environment import REGISTRY_ENV
# from MyInvertedPendulum import MyInvertedPendulumRLModel
# from xuance.environment import make_envs
# configs_dict = get_configs(file_dir="NewEnv.yaml")
# configs = argparse.Namespace(**configs_dict)
# REGISTRY_ENV[configs.env_name] = MyInvertedPendulumRLModel
# envs = make_envs(configs)
# Agent = KDQN_Agent(config=configs, envs=envs)













