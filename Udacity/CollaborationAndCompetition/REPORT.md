##  Report
### Project description
This project implements the Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments by Rowe et al using a modified two agent
tennis playing env from Unity as a test case. <br>

### Algorithmic goals and contributions of Rowe et al
- Traditional Q-learning (in the multi-agent case)difficulty is challenged
by an inherent non-stationarity of the environment, while
policy gradient suffers from a variance that increases as the number of agents grows.

- The Authors adapt the DDPG algorithm to the multi agent setting by proposing each agent has a
 centralised Q function (by centralised we mean that knowledge of the other agents' policies are needed by each agent in order to calculate its  own Q function ) . Additionally the have a decentralised actor network is used to take actions in the environment

### Learning Algorithm

#### Approach 1

At first I implemented MADDPG where the other agent had perfect knowledge of the the competitors
policies i.e All DDPG agents had access to all the competitor agent's **true actor and target actor policies**. This is the standard implementation of the MADDPG paper. In this approach  (as stated in *Algorithmic goals and contributions of Rowe et al*) , each agent has its own decentralised copy of a Q function estimating the future rewards(for the agent) given the states of **all the agents** (i.e. all agents has access to their competitors states tensors) and given the actions of **all the agents** (i.e. the agent has access to its competitors policy networks for inference of state tensors ).Each agent has the following
- A local actor network
- A target network
- A local critic network
- A target critic network
- A list of (competitor) DDPG agents where each agent has its own local and target actors ; local and target ciritcs

 . This resulted in fast learning once a reward signal was obtained. This approach was also
robust to many hyperparameters working under a wide variety of buffer sizes, learning rates and hidden unit widths. See CompetitorScores.png for the performance of this approach.

###### Code:
  In my code the standard MADDPG is instantiated as `maddpgagent = MADDPG(0,use_inferred_policies = True)`

#### Approach 2

My second approach relaxed the assumption of access to competitor policies and instead approximates them as suggested in section 4.2 of the paper. As stated in the paper:

*To remove the assumption of knowing other agents’ policies, as required in Eq. 6, each agent i
can additionally maintain an approximation of the other agents true policy. This approximate policy is learned by maximizing the log probability of agent j’s actions, with an entropy regularizer.*

 What is different from the first approach is that the target actions used to minimise the bellman squared error between the local and target Q functions come from the the *approximate actor network* rather than the target network of the competitor.

 Since we are acting in continuous space I choose to model the **inferred competitor policies** as being sampled
 from a multi-variate gaussian with 0 co-variance between agents in order to calculate log probabilities and the entropy of the agents needed for the regularization term. The **inferred competitor policies**  are trained in an online manner updating calculating the log loss with entropy regularization and backprop'ing the gradients before calculating the target Q value. As put more precisely in the paper :

*Note that Eq. 7 can be optimized in a completely online fashion: before updating Q centralized Q function, we take the latest samples of each agent j from the replay buffer to perform a single gradient step to update the parameters of the approximate network. Note also that, in the above equation, we input the action log probabilities of each agent directly into Q,
rather than sampling.*

 This method was highly sensitive to the choice in hyperparameters and the training process
was significantly less stable. The time taken to solve the environment (measured in number of episodes taken to learn an environment solving policy) was nearly 2X of the first approach.  See InferredCompetitorScores.png for the performance of this approach.

###### Code:
  In my code the MADDPG with approximate actors is instantiated as `maddpgagent_competitors = MADDPG(0,use_inferred_policies = False,competitor_policies = [])`

### Plot of Rewards

- See CompetitorScores.png
- See InferredCompetitorScores.png


#### Model Architecture

  Actor Network: <br>
  Vanilla MLP network

  * 1st hidden layer consists of 512 nodes<br>
  * 2nd hidden layer consists of 256 nodes<br>
  * Output layer is equal to the action size  <br>

  Critic  Network: <br>
  Vanilla MLP network

  * Input layer is equal to the full observation size = SUM(observationsize_for_all_agents) + SUM(actionsize_for_all_agents)<br>
  * 1st hidden layer consists of 512 nodes<br>
  * 2nd hidden layer consists of 256 nodes<br>


  #### Hyperparameters
  - BUFFER_SIZE = int(1e6)   replay buffer size <br>
  - BATCH_SIZE = 250         minibatch size <br>
  - GAMMA = 0.99             discount factor <br>
  - TAU = 1e-3               for soft update of target parameters <br>
  - LR_ACTOR = 3e-4          learning rate of the actor <br>
  - LR_CRITIC = 1e-3         learning rate of the critic  <br>
  - LR_INFERRED_POLICY = 2e-4        learning rate of the approximate competitor actor  <br>
  - NUM_AGENTS = 2 total number of agents in environment <br>


### Ideas for Future Work
- For exploration in this approach the Ornstein-Uhlenbeck process was used to generate noise for exploration. However it has been suggested that good ol Gaussian noise works just as well.
- Some form of prioritised replay from the buffer may lead to faster learning .

- Seeing if a similar method could be extended to other continuous space actor critic methods.

- Finding a more compact/dense representation for sequences of action and environment state pairs to aid in planning
