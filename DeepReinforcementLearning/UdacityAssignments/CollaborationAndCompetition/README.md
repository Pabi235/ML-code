[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"


# Solving the Tennis UnityML Environments using Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments

This repository contains an implementation of the  Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments paper using Deep Deterministic Policy Gradient agents to the Tennis UnityML Environment

### Paper
* "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments", Lowe et al, 2017<br>
https://arxiv.org/abs/1706.02275 <br>

[Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis)

![Trained Agent][image1]
### Description

#### Environment

Set-up: Two-player game where agents control rackets to hit a ball over the net.
Goal: The agents must hit the ball so that the opponent cannot hit a valid return.

Agents: The environment contains two agent with same Behaviour Parameters. After training you can check the Use Heuristic checkbox on one of the Agents to play against your trained model.

Agent Reward Function (independent):
+1.0 To the agent that wins the point. An agent wins a point by preventing the opponent from hitting a valid return.
-1.0 To the agent who loses the point.

Thus, the goal of each agent is to keep the ball in play.

Behavior Parameters:
Vector Observation space: 8 variables corresponding to position, velocity and orientation of ball and racket.

Vector Action space: (Continuous) Size of 2, corresponding to movement toward net or away from net and jumping. Each agent receives its own local observations

Visual Observations: None
Float Properties: Three
gravity: Magnitude of gravity
Default: 9.81
Recommended Minimum: 6
Recommended Maximum: 20
scale: Specifies the scale of the ball in the 3 dimensions (equal across the three dimensions)
Default: .5
Recommended Minimum: 0.2
Recommended Maximum: 5


The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
 - This yields a single score for each episode.
The environment is considered *solved* , when the average (over 100 episodes) of those scores is at least +0.5.

## Dependencies

pip install -r requirements.txt (Python 2), or pip3 install -r requirements.txt (Python 3)

### Environment setup

- Download the Unity Environment
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

    (For Windows users) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (For AWS) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will *not* be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.)


### Instructions
Run file `TennisCollaborationAndCompetition.ipynb` in a Jupyter notebook/lab to train the agent.

### Files
* `TennisCollaborationAndCompetition.ipynb` Instantiates the tennis environment, implements DDPG agents with two options
1. Knowledge of competitor policies
2. Approximating competitor policies
Implements Multi Agent training algorithm by Rowe et al. If MADDPG(use_inferred_policies = True)  approximates of competitor policies are implemented and the continuous actions are using to be sampled from a multi-variate Gaussian distribution   
* `networkforall.py` pytorch model architectures for actor , critic and inferred policies
* ` * pth` model weights for the various actor and critic methods
