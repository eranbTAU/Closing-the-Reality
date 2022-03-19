# MultiAgentEnv
Provided a MGR cars simulation implementation as described in the paper and a MARL training code. 
## simulator
The simulator is a modified [PettingZoo MPE](https://www.pettingzoo.ml/mpe/) multi-agent environment.

This is a parrallel settings env and the APIs are in GYM style.
A forward synamics model should be trained and placed in *maenv/mpe/_mpe_utils/forward_models/car/models* and it's filename should be passed with the arguments as a parameter in *main.py*

## MARL
*/MARL/DDPG/main.py* - training code example of a group of MGRs using a DDPG algorithm learning a single shared policy for all agents. 

