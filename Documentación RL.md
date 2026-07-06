














## Reinforcement Learning
## Documentación
## Proyecto: Slime Volley



Aimar Pagonabarraga Gallastegui - aimar.p.g@opendeusto.es
Jaime Etxebarria Ugarte - jaime.etxebarria@opendeusto.es









## INDEX

INTRODUCTION ..................................................................................................................... 3
ENVIRONMENT ADAPTATION AND INSTALLATION .......................................................... 4
IMPLEMENTED TECHNIQUES AND APPROACHES ........................................................... 5
INITIAL PPO TEST ............................................................................................................ 5
CURRICULUM LEARNING ............................................................................................... 6
LEARNING PROCESS STRUCTURE ......................................................................... 6
USED ALGORITHMS .................................................................................................. 8
TRAINING PROCESS ................................................................................................. 8
IMITATION LEARNING ................................................................................................... 12
## HOMOGENEOUS-COOPERATIVE-COMPETITIVE-SINGLE-POLICY-MARL ............... 14
CONCLUSIONS .................................................................................................................... 15




## INTRODUCTION

We decided to use the Slime Volleyball Gym Environment for the project. The environment implements
a 2 player game similar to Pong where each player must bounce the ball in their part of the field and
send it to the opponent. The reason for this choice was the fact that it allows for a great number of
experiments, can be used for multi-agent reinforcement learning, and the episodes are not excessively
long.

As  a  matter  of  fact,  among  those  experiments  previously  mentioned,  it  was  the  suitability  of  this
environment  for  curriculum  learning.  This  task  is  not  straightforward  to  be  learned,  but  it  can  be
decomposed into different, more simple tasks that can be learned sequentially and progressively, which
makes it a great use case to develop a curriculum learning strategy.

As stated before, it also allows for multiple multi agent variants. From this techniques several hypothesis
arised, among others the possibolities  of developing a cooperative variant for cooperative learning by
maintaining long matches and learning to dominate the physics of the environment; and afterwards,
turn into competitive to develop the defensive and aggressive skills to win games.



## ENVIRONMENT ADAPTATION AND INSTALLATION


In order to run the experiments, all that is needed is the “rl” environment we installed in class, with 2
extra dependencies: gym and

conda create --name volley python=3.10.18

conda activate volley

pip install -r requirements.txt

The  environment  was  originally  programmed  for  OpenAI  gym,  not  gymnasium,  and  the  experiments
used  stable-baselines, not stable-baselines3. Thus, the first thing we had to do was adapt it to the new
libraries,  which  was  not  too  complicated,  since  they  only changed  a  few  function  names  and
conventions. For rendering, we also had to adapt it to use pygame instead of the old rendering pipeline
from gym.


## IMPLEMENTED TECHNIQUES AND APPROACHES

## INITIAL PPO TEST
The first thing we did was test training the environment using simply PPO. The default environment
provided  in  the  repo  works  by  making  the  agent  play  against  a  pretrained  “expert”  policy  called
BaselinePolicy. This expert was developed in 2015 by David Ha using a simple genetic algorithm, and
will be used throughout the project as a standard in order to evaluate models.
This initial lasted 20M steps, so we could test the limits of the algorithm.


The results were decent, although they didn’t surpass the BaselinePolicy by much, peaking at around
0.1–0.2.  The  model  did  learn  quite  fast  how  to  avoid  dying  quickly  (the  episode  length  jumps  to  the
maximum 3000 steps at the start of the training process), but took significantly longer to learn how to
tie or beat the expert.




## CURRICULUM LEARNING
One  of  the  proposed  approaches  for  this  environment  is  curriculum  learning.  As  stated  in  the
introduction, when referring to the reasons why this environment was chosen, the task to be performed
in this environment needs to be learned in different stages with progressive levels of difficulty, so it turns
out to be a good opportunity for curriculum learning implementation.
For an agent to learn to play this game is not straightforward, it requires first for it to learn the basic
physics of the environment, to learn to bounce the ball and with which direction in order to return an
attack  efficiently.  And  afterwards  to  apply  that  knowledge  and  perfect  it  against  other  agents  with
different strategies and levels of difficulty, doing so increasingly in order to achieve our final goal that
would be to beat the base policy offered by the slimevolleygym library. As mentioned before, a great
advantage in the usage of this environment is the fact that it offers an already trained model that can
be used as a baseline to make comparisons and establish a goal to be achieved.

## LEARNING PROCESS STRUCTURE
Therefore,  the  curriculum  learning  solution  proposed  for  this  project  has  been  the  following.  It  is
structured in 4 phases that aim to follow the outline explained before (an initial period to learn basic
movements without much confrontation followed by incresign difficulty).
The first phase would correspond to that first coming together with the environment, in which the agent
learns what the game consists of and what it is supposed to do in order to achieve higher rewards.
Naturally, the agent cannot be faced with much difficulty since it does not yet understand what he is
doing and how it must respond and making it compete against an agent considerably better than him
will make him learn substantially slower or do not allow him to learn at all. That is the reason why it was
chosen that the opponent in this phase will be static (not moving at all), showing no confrontation against
the agent, allowing him to experiment and come to terms with the physics of the environment on its
own.
The first difficulty faced in this case is that the rewards of the environment are too sparse, the agent
only receives +1 or -1 when some of the players scores a goal (making the ball touch its opponents
field), which is not enough, or not straightforward enough, for the agent to conclude that it needs to,
firstly, move towards the ball, learn to efficiently jump to hit it and send it to the other side of the net to
score a point. This led to the need of implementing reward shaping in order to guide the learning of the
agent. Initially two new minimal rewards were added: first a distance reward, so the agent will receive
a  bonus  inversely  proportional  to  the  distance  between  him  and  the  ball;  and  another  reward  when
touching the ball.
However, this turned out to be too vague to work effectively, since the agent would learn to stay near
the net, which is where the kick-off happens, wher he will get the most reward only by staying still. This
clearly is unhelpful, so the logical thing to do was to change the distance reward to only be given if the
distance to the ball was reduced compared to the one in the previous state. So a variable delta would
be calculated as the change in euclidean distance with respect to the ball and then multiplied by a factor
of 0.05 to be given as a reward. This factor is necessary in order not to make this reward eclipse the
other, maybe more important rewards, such as kicking the ball and, obviously, scoring the goal.
Once  the  basic  physics  were  controlled  and  the  agent  had  learned  how  to  move  and  bounce,  the
moment came for the agent to perfect those movements and improve the quality of its game. The first
challenge was to fully manage the kick-off, which resulted in a negative score, in most of the cases in
which it was aimed at the agent. So in order to overcome this difficulty a wrapper was applied in order
to make the service be always aimed at the agent, this way it must mandatorily learn to properly respond


to a kick-off or it will lose all the points. This turned out to be helpful to achieve that wanted control of
the kick-off and in order to presumably increase the difficulty the opponent was made random, instead
of static. Nevertheless, this opponent was not as effective as expected, since the probability of an agent
performing random actions to not only bounce the ball but also to kick it and send it to the agent's field
was minimal, if not none. Basically, it was like playing against the static opponent from the previous
phase and the task did not seem difficult enough for the agent to improve consistently, it needed being
confronted by a real opponent, someone that will return the attacks, or at least some of them, and that
will represent an increase in the difficulty compared to Phase 1.
Consequently, the opponent was updated to follow a very simple heuristic: move towards the ball (if the
x coordinate of the ball is to the left, move left, if it is to the right, move right) and when the ball is found
within  a  reachable  distance,  jump.  This  distance,  unlike  in  Phase  1,  is  calculated  as  the  squared
distance in order to accelerate computation. Though it may appear like an extremely simple heuristic
for a game that, is not remotely as easy as it seems to be  for a human to play, this heuristic serve a
great  propose,  since  it  was  able  to  return  most  of  the  balls  sent  by  the  agent,  making  it  possible  to
maintain  longer  rounds  and  showing  him  what  having  a  real  opponent  could  be  like.  These  are  the
aspects  changed  with  respect  to  the  previous  phase,  however,  the  dense  rewards  have  been
maintained, as they help improve the game strategy of the agent without letting him forget the the most
important thing is to bounce the ball, send it back to the other and, above all, hit it not to let it touch the
floor of the his field.
After having gone through Phase 2 of this curriculum learning approach, it is believed that the agent is
ready to face real threats in the field by fighting contenders of similar skills. On this account, this third
phase will consist in an iterative self-play in which initially both the agent and the opponent are loaded
with  the  same  model,  the  best  achieved  on  Phase  2,  and  with  certain  frequency  the  opponent  is
updated. For this update to happen, within that period an evaluation callback is used to evaluate the
current model and store the best model reached, when that period is finished that best model is stored
in a pool of opponents as a new opponent generation and it is loaded as the new opponent of the model.
This way, with that established frequency the opponents are updated and the difficulty the agent has to
confront  grows  progressively.  Self-training  obliges  the  agent  to  go  up  against  its  own  developed
attacking skills and learn how to improve the defensive ones, especially because the agent will use its
own  weak  spots  in  order  to  attack,  which  will  become  a  challenge  when  comingup  against  the  next
generation.
Finally,  the  agent  reaches  the  fourth  phase  of  the  learning  process.  It  is  supposed  to  have  already
learned the basics, faced its own weaknesses and perfected its offensive skills, so now should be able
to confront a proper opponent with developed expertise on the arena, as is the case of the base policy
of the environment. Before it was stated that training an agent with no prior knowledge of the game
could be counterproductive, this is not the case, however, of the current agent, which already knows its
way through the field and how to defend itself against attacks of certain difficulty.
Initially, the idea for this phase was constant training against the baseline. This pretrained model was
considered a representation of expertise on the matter and, therefore, the final objective to achieve was
to end up playing as well or, if possible, even better than it. Nonetheless, after having trained for several
steps with against this model and having accomplished better performance than the baseline policy (a
reward of around 0.2, meaning the agent wins slightly more matches than he loses against this policy),
it  was  considered  that,  even  if  the  baseline  policy  had  been  considered  the  best  model  adn  a  good
reference to be compared with, training to develop a game strategy by just overfitting the behaviour of
one unique opponent was not a good practice, having also in mind that maybe it could be forgetting
some  of  the  defensive  techniques  learned  before  because  they  might  have  not  been  needed  when
playing against the baseline.
For this reason, it was concluded that the best solution was to divide this phase into two periods: a
warm-up period in which the agent will train only against the baseline policy to get accostumed to it;


and then a “League Training” period, in which a pool of opponents would be created against which the
agent will fight randomly in each of the consequent episodes. This League Training is extrictly similar
to the iterative self-play of the previous phase, but here the opponents will not change sequentially, but
will be chosen randomly for every next episode. In this random selection of opponents, however, will
also be included the baseline policy so that the agent can keep training with it and can keep improving
its strategy. After all, the baseline is the adversary it is going to be brought face to face with in every
evaluation callback during the training. In summary, after training against the basleine, it will battle it
with a probability of 0.3 and the rest of the time one of the agents stored in the league pool, everyone
with the same probability of being chosen.

## USED ALGORITHMS
The algorithms used to train agents through these curriculum learning phases were PPO and DQN.
Both of these algorithms are an accurate choice to try to solve this environment, whose observation
space is continuous, but action space is discrete.
On the one hand, PPO is an On-Policy algorithm (that is, the policy to be improved is the same used in
training to collect experiences) whose aim is to directly learn the probability distribution function of the
action. This algorithm is also of type Actor-Critic, meaning it has two neural networks, or one with two
outputs: the Actor is the part that tries to predict the probability distribution for the actions in each state
and learns to maximize the expected reward based on these probabilities; the critic, alternatively, tries
to predict the expected future reward of a certain state by minimizing the error between the expected
reward and the really obtained one. Among the reasons why PPO is an appropriate choice for this task
one  can  find  the  following  ones:  firstly,  its  suitability  for  continuous  spaces  should  be  highlighted,
because, though the actions are discrete, the input is continuous and the neural networks in PPO map
little changes in the position to probability distributions over the actions, generating very robust policies;
equally important is its intrinsic estocasticity, which allows the agent not to be completely predictable,
making it more difficult for the opponent to overfit into a deterministic behaviour; and, finally, its stability
in training, since the mechanism of PPO guarante that the policy updates will be within a trust region
ensuring a monotonous and controlled learning.
On the other hand, DNQ is an Off-Policy (meaning that the policy to be improved is not the same as
used when training) algorithm whose objective, unlike PPO, is not to learn what to do, but to predict the
value  of  being  in  a  certain  state  and  performing  a determined  action.  This  means,  in  mathematical
terms,  to  try  to  approximate  the  function  of  the  optimal  action  value,  aimed  at  minimizing  the  error
between the current reward and the temporal difference target. Regarding the reasons why it suits this
environment: its optimality for discretized action spaces, due to the fact that calculating the maximum
reward of a play is considerably easy with a discrete action space, and too complex computationally in
a  continuous  one;  and  the  use  of  bootstrapping,  which consists  in  making  a  prediction  based  on  a
previous prediction and allows for a future reward to efficiently spread back in time.
It  must  be  mentioned  that,  though  DQN  also  works  with  discrete  action  spaces,  in  this  case  the
environment offer 3 multi-binary buttons to play, which can be combined amid them, so a discretization
wrapper  was  applied  to  map  all  the  possible  binary-button-combinatios  into  6  absolutely  discrete
actions: do nothing, move left, right, jump, move left and jump and move right and jump.
## TRAINING PROCESS
This section offers a detailed analysis of the training process for each algorithm through each one of
the 4 phases of the curriculum learning implementation. The four phases were trained separately, in
each one of them loading the best model achieved in the previous one to continue the training, with the
exception of, naturally, the first one, which starts from scratch. The first two phases were trained for 2M


steps, based on the fact that around 1M to 1.5M steps the training already reached the plateau so more
training  would  not  ensure  better  results.  The  third  and  fourth  phases  were  trained  for  3M  and  5M
timesteps respectively to allow for more development of the game strategy thanks to the self-play and
league training.

Phase 1 training

These two graphs correspond to the episode mean reward and the mean reward during evaluation of
both agents (PPO and DQN) during training of the first phase of the curriculum learning approach. As
stated in the description of the phase logic, in this case both agents start from scratch against a static
opponent. The bigger challenge for the agents here was to learn basic physics and confront the serve
when it was its turn.
Obviously the opponent represented no difficulty nor danger for the agent because could not respond
to  the  attacks,  that  is  the  reason  why  rewards  that  high  can  be  seen  in  the  graphs.  In  the  original
environment the rewards were dense and discrete, only the amount of goals scored during the episode
(match), so the maximum and minimum reward would be 5 and -5 respectively. In this section, however,
rewards higher than 5 can be seen as a result of the reward shaping applied by the wrapper of this
section, rewarding both getting nearer to the ball and hiting it.
As  it  can  be  seen,  PPO  easily,  rapidly  learns  the  basics  and  reaches  high  rewards  by  beating  the
opponent completely (from the point of 800K onwards it seems to win all the episodes by scoring all the
goals).  Nonetheless,  that  does  not  seem  to  be  the  case of  DQN.  This  could  be  understood  by  the
difference in the way each algorithm learns, as explained before, in the case of DQN it is considerably
difficult to build a function to assign a value for each action in each state-action pair. However, PPO
learns a policy over the actions, basically learning to follow the direction of the gradient (to move towards
the  ball),  which  is  a  far  easier  task  for  a  neural  network.  In  the  same  way,  DQN  uses  a  greedy
exploration,  which  can  result  in  more  serious  mistakes  leading  to  a  more  more  chaotic  learning;
adversely, PPO explores based on entropy, which means that the agent will explore in states in which
he is more uncertain.
Nonetheless, it can also be noticed that on the evaluation the DQN agent almost reaches the same
maximum as the PPO agent. This matters because in the next phase the best model achieved will be
the one used to start training.




Phase 2 training

In this second phase the opponent would represent a bigger difficulty compared to that of the previous
one,  but  not  so  severe  that  the  agent  would  not  learn  to  beat  it  easily  and  earn  high  rewards  in  a
relatively  short  time.  Similar  to  the  situation  seen  in  the  previous  graphs,  here  the  PPO  agent  also
seems to overperform compared to the DQN agent.
Another  possible  explanation  for  this  underperformance  could  be  the  instability  caused  by  the
bootstrapping mechanism from initialized values, which updates its predictions based on its own future
predictions, which may be incorrect in the first place. This, in conjunction with the maximization operator,
that makes the algorithm optimistic by only choosing the highest value, can make early errors to be
propagated backward. However, both agents end up learning and the best models in both cases win
against the basic-heuristic-based opponent.

Phase 3 training

Regarding the third phase of the curriculum learning, something curious happens. PPO seems to too
easily learn to beat its opponent, which is none other than himself. This seems extrange because the
initial hypothesis was that both agents start with rewards next to 0, since they will draw most of the
times with theirselves, which is the case of DQN, but definitely not that of PPO.
A possible explanation for this could be that, even if the models have learned to play the basics in the
previous  phases,  they  carry  many  weak  spots.  These  opponents  no  longer  have  the  capability  to
improve,  but  the  PPO  agent  can  rapidly  detect  those  weaknesses  and  overfit  them  to  defeat  the
opponent quite plainly. The interesting thing is that, after having learned how to drastically attack those
weaknesses, then that improved version of itself is going to become the opponent, who is going to try
to  use  those  strategies  on  him,  obliging  him  to  improve  its  defensive  skills  and  erradicate  those
vulnerabilities.


It is worth mentioning that this part was initially set to be trained for 2M steps only, and the opponent
update frequency was too low (once every 400K steps). This resulted in the agent losing too much time
computing against its older version, which were consequently worse, and beating them too easily, so
he would learn very few new strategies. That is why, it was decided to increase the update frequency
considerably and train it for longer (3M), in order to have more opportunities for self improvement.

Phase 4 training

In this final phase, the results were more or less as expected. In the initial warm-up period against the
baseline policy the agents clearly performed worse (up to 250K steps). This is expected as the baseline
policy  is  considered  an  expert  on  the  matter  and  not  easily  defeated.  However,  it  can  be  seen  that
during that period the agents improved, meaning that they could learn to develop a strategy against an
opponent of such level. Afterwards, when competing against the opponents on the leagues (previos,
and  presumably  worse,  versions  of  themselves)  the  mean  reward  goes  up  as  is  normal,  since  it
represents the mean between some easier and other more difficult (battling the baseline) matches.
Here the overperfomance of the PPO compared to DQN does not matter as much, since it could mean
the it is harder for the DQN agent to defeat its opponents because they might even be better than those
of PPO. However, reality is quite plain when looking at the evaluation graph, in which it is shown that
when playing against the baselin (the final acid test) PPO performs better. It should be highlighted that
the final, and best, reward obtained by the PPO agent is 1 against the baseline, meaning it is able to
consistently beat the default expert model by one goal. This can be compared with the initial test trained
for 20M steps, which reached a final and best reward of 0.1. This can be taken as a signal that the
curriculum learning implementation works effectively, since a higher mean reward was achieved and
with almost half of the training required.








## IMITATION LEARNING

Another technique we applied was imitation learning. Specifically, we tried Behavioral Cloning using the
imitation library. The goal was to create an agent that copied the behaviour of the BaselinePolicy, which
is  an  “expert”  pretrained  policy  included  in  the  SlimeVolley  environment.  This  required  some
modification  to  the  environment,  since  the imitation library  requires  the  usage  of  vectorized
environments and only works with floats.

The process of behavioral cloning is as follows:
- Create a vectorized environment
- Use those environments with the expert policy to sample some episodes (e.g 100 episodes)
- Using those sampled “Trajectories” (set of observations, actions and rewards), calculate the
simpler “Transitions” (batches of observation-action-observation-done quadruples).
- Train  the  model  (FeedForward32Policy)  using  the  Behavioral  Cloning  algorithm  and  the
calculated transitions. This is done by feeding the model an observation and trying to minimize
the difference between the predicted and expected outputs (supervised learning).
- Evaluate the trained model.

The results are quite good, getting evaluations around the -0.6 – 0.2 mark (BaselinePolicy “expert” is 0,
minimum -5, maximum +5), with just 50 seconds of training.

In the tensorboard charts we can see how the loss plummets and the probability of choosing the correct
action rises to about 75%. Training for longer than 5 epochs or more than 100 episodes of data did not
result in any significant improvements (+-3%), so ~75% seems to be the limit.







## HOMOGENEOUS-COOPERATIVE-COMPETITIVE-SINGLE-POLICY-MARL
Another technique we applied was multi-agent reinforcement learning, where the agent learns against
a copy of itself instead of playing against the “expert”. In this case, it is homogeneous, since both agents
have the same role, cooperative-competitive, since one can gain more than the other but they can also
help each other get higher rewards by staying alive for longer (passing the ball instead of throwing it
trying to score), and only has one training policy instead of multiple indepent ones.
In this case the experiment we wanted to do was to see whether the policy would develop so the agents
“helped” each other survive for longer, by testing 2 reward functions: one that rewards scoring points,
and one that adds a reward for surviving. Looking at the mean episode length of the evaluations in the
tensorboard charts, the results are clear. The policy does indeed develop a bias towards staying alive,
meaning the 2 agents cooperate to avoid having short episodes, favoring instead a “passing the ball”
strategy.




We suspect that the reason they prefer to cooperate than to compete is the following: an agent (agent
A) who “betrays” the other, initially has a greater reward than the one that doesn’t (agent B), but after
winning for a while, the opponent (agent B) will be updated to have the same policy as the first one
(agent A). This will level the playing field again (both try to score instead of cooperate), meaning they
are back at a point where only striving for longer games translates to consistently higher rewards.



## CONCLUSIONS

In conclusion, the SlimeVolley gym presented an interesting challenge which was solvable in multiple
ways. Curriculum learning achieved better results and in less time than the simpler “brute force” PPO,
surpassing the BaselinePolicy, which the standard PPO training could barely do. In this case, it must
also be mentioned that when solving these tasks, a great deal of attention must be paid to the internal
logic of the algorithms that are going to be used, since this could limit the development of the agent's
learning depending on which approach the algorithm uses to learn. This could be seen as, or at least is
the most reasonable explanation found for the divergence in the learning and training process between
both algorithms, as a symbol that depending on the task the approach to be used is fundamental. In
this case, as described in the corresponding section, the results, especially those of the PPo agent,
were quite promising as it reached a value of 1 when being evaluated against the baseline policy.
With regards to the imitation learning implementation with Behavioral Cloning, this also resulted in a
success, enabling very fast training with decent results, which did not surpass the expert/teacher, as
expected, but showed remarkable outcomes with very little compute time, reaching average rewards of
-0.6 – 0.2.  The  MARL  experiment  showed  how  the  agents  learned  to  cooperate  by  increasing  the
episode length.





