# MultiAgentEnv
Provided a MGR cars simulation implementation as described in the paper and a MARL training code. 
## simulator
The simulator is a modified PettingZoo multi-agent environment.
This is a parrallel settings env and the APIs are in GYM style.
A forward synamics model should be learned and placed in *maenv/mpe/_mpe_utils/forward_models/car/*
## MARL
/MARL/DDPG/main - training code example of a group of MGRs using a DDPG algorithm learning a single shared policy for all agents. 

