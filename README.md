#### What's included:
- */Car* - MGR CAD and data based forward-dynamics model training resources.
- */Fish* - MUR CAD and data based forward-dynamics model training resources.
- */GAN* - Synthetic forward-dynamics model training resources.
- */MultiAgentEnv* - GYM style simulation environement with DDPG training algorithm implementation

## Real-to-Sim-to-Real: Learning Models for Homogeneous Multi-Agent Systems
following the paper, we share here the used forward dynamics model of the robots, the hardware CAD and specifications and the resources of the multi-robot simulation for RL traning, as used and described in the paper.   

#### abstract
Training Reinforcement Learning (RL) policies for a robot requires an extensive amount of data recorded while interacting with the environment. Acquiring such a policy on a real robot is a tedious and time consuming task. It may also cause mechanical wear and pose danger. This is more challenging in a multi-agent system where individual data is required from each agent. While training in simulations is the common approach due to efficiency and low-cost, they rarely describe the real world. Consequently, policies trained in simulations and transferred to the real robot usually perform poorly. In this letter, we present a novel real-to-sim-to-real framework to bridge the reality gap for homogeneous multi-agent systems. First, we propose a novel deep neural-network architecture termed Convolutional-Recurrent Network (CR-Net) to simulate agents. CR-Net includes convolutional, recurrent and fully-connected layers aimed to capture the complex transition of an agent. Once trained with data from one agent, we show that the CR-Net can also accurately predict motion of other agents in the group. Second, we propose to invest a limited amount of real data from one agent in a generative model. Then, training the CR-Net with synthetic data sampled from the generative model is shown to be at least equivalent to real data. The generative model can also be disseminated along with open-source hardware for easier usage. Using the models, we build a simulation that is based solely on data from one agent. We show experiments on ground and underwater vehicles in which multi-agent RL policies are trained in the simulation and successfully transferred to the real-world.

#### CR-Net model
![CR-Net](https://user-images.githubusercontent.com/77546342/154935368-9ecf6c01-dff2-49f4-9b17-9a304920b26e.png)

#### Sim and real for group of MGRs 
<img src="https://github.com/eranbTAU/Closing-the-Reality/blob/6048910f50c07263f16add6c11f7146b5621c00c/imgs/car_sim_2_slow.gif" width="300"> <img src="https://github.com/eranbTAU/Closing-the-Reality/blob/217205c9de04b2cd25f82ee4a3992bbd02d17044/imgs/car_real_2_slow.gif" width="300">

#### Sim2Real for a couple of MURs
<img src="https://github.com/eranbTAU/Closing-the-Reality/blob/231271f725f2d3eeb031b193c90412f0b5684f0c/imgs/fish_1.gif" width="300"> <img src="https://github.com/eranbTAU/Closing-the-Reality/blob/231271f725f2d3eeb031b193c90412f0b5684f0c/imgs/fish_2.gif" width="300">
