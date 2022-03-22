# MultiAgentEnv
Provided a MGR cars simulation implementation as described in the paper and a MARL training code. 
## simulator
*/maenv*

The simulator is based on a modified [PettingZoo MPE](https://www.pettingzoo.ml/mpe) multi-agent environment.

This is a parrallel settings env and the APIs are in GYM style.
A forward synamics model should be trained and placed in *maenv/mpe/_mpe_utils/forward_models/car/models* and it's filename should be passed as an arguments in *main.py*.

<img src="https://github.com/eranbTAU/Closing-the-Reality/blob/07cd65353f3eb5c50477072869f8c6da20794ad7/multiAgentEnv/sim2.png" width="600">

## MARL
*/MARL/DDPG/main.py* - training code example of a group of MGRs using a DDPG algorithm learning a single shared policy for all agents. 

