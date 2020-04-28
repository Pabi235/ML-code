<img src="https://camo.githubusercontent.com/7ad5cdff66f7229c4e9822882b3c8e57960dca4e/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f766964656f2e756461636974792d646174612e636f6d2f746f706865722f323031382f4a756e652f35623165613737385f726561636865722f726561636865722e676966">


# Continous control for robotic arms in the Reacher env



### To run:
- To currently run please open the .ipynb file and run all the cells


### Requeriments:
ipykernel==4.9.0
ipython==6.5.0
ipython-genutils==0.2.0
ipython-sql==0.3.9
ipywidgets==7.0.5
jupyter==1.0.0
jupyter-client==5.2.4
jupyter-console==6.0.0
jupyter-core==4.4.0
matplotlib==2.1.0
nbconvert==5.4.0
numpy==1.12.1
scikit-learn==0.19.1
scipy==0.19.1
seaborn==0.8.1
torch==0.4.0
torchvision==0.2.1
tornado==4.5.3
tqdm==4.11.2
unityagents==0.4.0
urllib3==1.22
webencodings==0.5
websockets==4.0.1
widgetsnbextension==3.0.8


## The problem:
* Set-up: Double-jointed arm which can move to target locations.
* Goal: The agents must move its hand to the goal location, and keep it there.
* Agents: The environment contains 10 agent with same Behavior Parameters.
* Agent Reward Function (independent):
    * +0.1 Each step agent's hand is in goal location.
* Behavior Parameters:
    * Vector Observation space: 26 variables corresponding to position, rotation, velocity, and angular velocities of the two arm Rigidbodies.
    * Vector Action space: (Continuous) Size of 4, corresponding to torque applicable to two joints.
    * Visual Observations: None.


## The solution:
I implemented the DDPG and A2C algorithms for this problem