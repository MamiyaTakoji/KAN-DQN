通过训练一个简单的智能体比较了使用MLP和KAN的效果，运行基于[xuance](https://xuance.readthedocs.io/en/latest/index.html)的1.2.1版本，KAN的实现参考了[efficient-kan](https://github.com/Blealtan/efficient-kan)，1.py训练了使用MLP的智能体，2.py训练了使用KAN的智能体，环境在BuyEnv1.py中实现。

为了公平比较，两个智能体的参数量大致是相同的，但是2.py的运行时间仍然是1.py的6倍左右，两个智能体都能差不多学到最优动作（MLP离最优动作差一点，KAN完全学会，但是可能换一个种子就是反过来的结果）

训练的记录以及模型可以通过log文件夹以及model文件夹直接获取

（translated by AI)

A simple agent was trained and compared using an MLP and KAN, running on version 1.2.1 of [XuanCe](https://xuance.readthedocs.io/en/latest/index.html). The implementation of KAN referred to [efficient-kan](https://github.com/Blealtan/efficient-kan). The agent using MLP was trained with 1.py, while the agent using KAN was trained with 2.py, and the environment was implemented in BuyEnv1.py.

To ensure a fair comparison, the number of parameters for both agents was kept roughly the same. However, the runtime of 2.py is still about 6 times that of 1.py. Both agents can learn actions close to optimal (the MLP is slightly less accurate than the optimal action, whereas KAN learns it completely, although with a different seed, the result might be reversed).

The training records and models can be directly obtained from the log folder and model folder, respectively.