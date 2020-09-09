# DQN-HER
This repository contains the Pytorch implementation of Deep Q-Networks with hindsight experience replay to solve the bit flip problem consisted of 25 bits.  
## Dependencies
numpy == 1.18.2
torch == 1.2.0
## Hyper-parameters
> n_bits -> Number of bits
> lr -> Learning rate
> gamma -> Discount factor
> k_future -> 

|  Parameter  |  Value  |
| :---------: | :-----: |
|   n_bits    |   25    |
|     lr      |  1e-3   |
|    gamma    |  0.98   |
| memory_size | 1000000 |
| batch_size  |   128   |
|  k_future   |    4    |

<p align="center">
<img src="https://user-images.githubusercontent.com/32295763/77784113-e5e6ee00-7051-11ea-9359-b6feb30a3134.png" >
</p>
