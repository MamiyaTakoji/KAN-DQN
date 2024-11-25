# -*- coding: utf-8 -*-
"""


@author: Mamiya


tensorboard --logdir ./logs/dqn/MLP


"""

import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from BuyEnv1 import BuyingRLModel
from xuance.environment import make_envs
from xuance.torch.agents import DQN_Agent
import shutil
import os
logPath = "logs/dqn/MLP"
if os.path.exists(logPath) and os.path.isdir(logPath):
    shutil.rmtree(logPath)
#shutil.rmtree("models")
configs_dict = get_configs(file_dir="BuyEnv1.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = BuyingRLModel
envs = make_envs(configs)
Agent = DQN_Agent(config=configs, envs=envs)
Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.


#%%测试智能体
from BuyEnv1 import TestEnv2
import numpy as np
IsFinish = False
BuyTest = TestEnv2()
turnNum = 0
while IsFinish is not True:
    turnNum += 1
    state = BuyTest.GetObservation()
    action = int(Agent.policy(state)[1])
    action = BuyTest.GetAction(action)
    IsFinish = BuyTest.OneStepOperate(action)
    print(state*200)