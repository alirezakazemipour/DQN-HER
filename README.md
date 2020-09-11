# DQN-HER
This repository contains the Pytorch implementation of Deep Q-Networks with hindsight experience replay to solve the bit flip problem consisted of 25 bits.  
The number of bits can be changed; 25 and more are considered to be slightly difficult challenges.   

## Dependencies
- numpy == 1.18.2  
- torch == 1.2.0  
## Hyper-parameters
> n_bits: Number of bits  
> lr: Learning rate  
> gamma: Discount factor  
> k_future:  Number of episode for _future_ hindsight strategy  

|  Parameter  |  Value  |
| :---------: | :-----: |
|   n_bits    |   25    |
|     lr      |  1e-3   |
|    gamma    |  0.98   |
| memory_size | 1000000 |
| batch_size  |   128   |
|  k_future   |    4    |

## Result
<p align="center">
<img src="https://user-images.githubusercontent.com/32295763/77784113-e5e6ee00-7051-11ea-9359-b6feb30a3134.png" >
</p>  
According to the plot when the problem is solved, at the worst case scenario, it takes 12 bits to be flipped by the agent in order to achieve the real, desired goal.  

## Reference
1. [_Human-level control through deep reinforcement learning_, Mnih et al., 2015](https://www.nature.com/articles/nature14236)  
2. [_Hindsight Experience Replay_, Andrychowicz et al., 2017](https://arxiv.org/abs/1707.01495)  
