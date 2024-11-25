import numpy as np
from xuance.environment import RawEnvironment
from gym.spaces import Box,Discrete
class BuyingRLModel():
    def __init__(self,env_config):
        super(BuyingRLModel, self).__init__()
        self.env_id = env_config.env_id  # The environment id.
        low = np.array([0,0,0]);high = np.array([700,700,700])
        self.observation_space = Box(low = low, high = high, shape=(3, ))  # Define observation space.
        #self.action_space = Discrete(9)
        self.action_space = Discrete(11)
        self._current_step = 0  # The count of steps of current episode.
        self.TestEnv2 = TestEnv2()
        self.max_episode_steps = 15
    def reset(self,**kwargs):
        self._current_step = 0
        self.TestEnv2.reset()
        return self.TestEnv2.GetObservation(),{}
    def step(self,action):
        self._current_step += 1
        action = self.TestEnv2.GetAction(action)
        terminated = self.TestEnv2.OneStepOperate(action)
        if terminated == True:
            rewards = (-np.abs(self.TestEnv2.waterNum-173)-np.abs(self.TestEnv2.foodNum-217))
        else:
            rewards = 0
        #rewards = GetReward(self.TestEnv2.waterNum, self.TestEnv2.foodNum)
        observation = self.TestEnv2.GetObservation()
        truncated = False if self._current_step < self.max_episode_steps else True
        if truncated ==True:
             rewards = (-np.abs(self.TestEnv2.waterNum-173)-np.abs(self.TestEnv2.foodNum-217))
        info = {}
        return observation,rewards,terminated,truncated,info
    def render(self, *args, **kwargs):  # Render your environment and return an image if the render_mode is "rgb_array".
        return np.ones([64, 64, 64])
    def close(self):  # Close your environment.
        return
class TestEnv2():
    def __init__(self):
        self.waterNum = 0
        self.foodNum = 0
        self.maxMonery = 10000
        self.maxHold = 1200
        self.waterCost = 5
        self.waterWeight = 3
        self.foodCost = 10
        self.foodWeight = 2
        self.turnNum = 0
    def OneStepOperate(self,action):
        self.turnNum += 1
        if action[0]==0 and action[1]==0:
            return True
        RestMoney = self.maxMonery - self.waterNum*self.waterCost - self.foodNum*self.foodWeight
        RestHold = self.maxHold - self.waterNum*self.waterWeight-self.foodNum*self.foodWeight
        addFoodNum = action[1]
        addWaterNum = action[0]
        addMoney = addWaterNum*self.waterCost + addFoodNum*self.foodCost
        addWeight = addWaterNum*self.waterWeight + addFoodNum*self.foodWeight
        #目前只考虑单独买水或者单独买食物的情况
        IsFinish = False
        if RestMoney <= addMoney or RestHold <= addWeight:
            addFoodNum = int(min(np.floor(RestMoney/self.foodCost),
                             np.floor(RestHold/self.foodWeight)))
            addWaterNum = int(min(np.floor(RestMoney/self.waterCost),
                             np.floor(RestHold/self.waterWeight)))
            IsFinish = True
        self.foodNum = self.foodNum + addFoodNum
        self.waterNum = self.waterNum + addWaterNum
        return IsFinish
    def GetAction(self,inputAction):
        Dic = {0: [20, 0], 1: [10, 0], 2: [5, 0], 3: [1, 0],
                4: [0, 20], 5: [0, 10], 6: [0, 5], 7: [0, 1], 
                8: [0, 0], 9: [50, 0], 10: [0, 50]}
        #Dic = {0: [0, 0], 1: [0, 50], 2: [50, 0]}
        return Dic[inputAction]
    def reset(self):
        self.waterNum = 0
        self.foodNum = 0
        self.turnNum = 0
    def GetObservation(self):
        return np.array([self.waterNum/200,self.foodNum/200,self.turnNum/15])