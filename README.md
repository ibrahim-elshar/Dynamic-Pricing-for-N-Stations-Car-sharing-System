# Dynamic Pricing for N-Stations Vehicle Sharing System

### IE3186 ADP project

Suppose that a vehicle sharing manager is responsible for setting the rental price for the vehicles at the beginning of each period in a finite planning horizon consisting of T periods of equal length.
We study a N-stations car sharing system. The goal is to optimize the prices to set for renting a car at each of the N stations.

We formulate the problem as a Markov decision process (MDP). Due to both the well known curse of dimensionality and the fact that the actions are continuous, the MDP cannot be solved exactly by Dynamic Programming. This motivates us to work with Approximate Dynamic Programming (ADP) algorithms to find a good solution to this pricing problem.

We propose a new ADP algorithm, *DP Policy Mixing (DPPM)* and compare it to the state-of-the-art algorithm Deep Deterministic Policy Gradients (DDPG) reinforcement learning (RL) algorithm, proposed by Lillicrap et al . (2015) to solve RL problems with continuous actions.

DPPM approximates the N-Stations problem by a N two-stations problems and solves each of them exactly by dynamic programming. 
To mitigate the problematic continuous pricing decision vector in the DP, we use the expected demand  instead of the price as the decision vector and consider only its discrete values.
This discretization helps us to solve the DP for each of the N problems exactly by backward induction. 
The optimal policy for each station in the two-station problems is then used to generate a policy for the N-station problem. Preliminary experimental results show that the resulting policy is good by itself.
We try however to improve it by an Evolution Strategy (ES) algorithm that takes the DP policies as input to come up with a policy.


#### Code:

For adp_gym_carsharing environment please refer to the readme in the adp_gym_carsharing folder.

For DPPM algorithm first run DPPM\DP\DP.py to obtain the DP policy for each station and the combined multi_station_policy
to use evolution strategy run DPPM\ES\run_cs.py

For DDPG algorithm run ddpg_adp_carsharing.py in DDPG/ddpg_adp_carsharing

