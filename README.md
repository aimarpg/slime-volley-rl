# Slime Volley Reinforcement Learning

A project exploring various Reinforcement Learning techniques within the Slime Volleyball Gym Environment.

**Authors:**
- Jaime Etxebarria Ugarte (jaime.etxebarria@opendeusto.es)
- Aimar Pagonabarraga Gallastegui (aimar.p.g@opendeusto.es)

## Introduction

This project utilizes the [Slime Volleyball Gym Environment](https://github.com/hardmaru/slimevolleygym) to train reinforcement learning agents. The environment simulates a 2-player game similar to Pong, where each player must bounce the ball on their side of the field and send it to the opponent.

We chose this environment because it offers a great testing ground for multi-agent reinforcement learning (MARL), curriculum learning, and imitation learning, with relatively short episodes and a pre-trained expert policy (`BaselinePolicy`) to use as a benchmark.

## Environment Adaptation & Installation

The environment was updated from standard OpenAI `gym` to `gymnasium` and adapted to use `pygame` for rendering instead of the deprecated gym rendering pipeline. The experiments were run using Stable Baselines 3.

To run this project, you can set up a conda environment:

```bash
conda create --name volley python=3.10.18
conda activate volley
pip install -r requirements.txt
```

## Implemented Techniques

### 1. Initial PPO Test
We started by testing the default environment against the `BaselinePolicy` using Proximal Policy Optimization (PPO) for 20M steps. The agent learned how to avoid losing quickly and eventually managed to tie or slightly beat the expert, but it took a significant amount of time.

### 2. Curriculum Learning
We implemented a 4-phase curriculum learning approach to guide the agent from learning basic physics to competing against experts:

- **Phase 1 (Basic Physics):** The agent plays against a static opponent with reward shaping (distance to ball, hitting the ball).
- **Phase 2 (Basic Heuristics):** The opponent is updated to a simple heuristic agent that moves towards the ball and jumps. Dense rewards are maintained.
- **Phase 3 (Iterative Self-Play):** The agent trains against previous versions of itself. Opponents are periodically updated from a pool of best past models, forcing the agent to improve both offensive and defensive skills.
- **Phase 4 (League Training & Expert):** The agent warms up against the `BaselinePolicy` and then plays against a random pool of past versions and the baseline expert.

**Results:** PPO significantly outperformed DQN, reaching a final reward of 1 (consistently winning by one point) against the `BaselinePolicy` in almost half the training time compared to the initial brute-force approach.

### 3. Imitation Learning (Behavioral Cloning)
We utilized the `imitation` library to train an agent to mimic the `BaselinePolicy` using Behavioral Cloning. 
After extracting 100 episodes of trajectories from the expert, the agent achieved around 75% accuracy predicting the expert's actions in just 50 seconds of training. Evaluations showed performance ranging between -0.6 and 0.2 against the expert.

### 4. MARL (Homogeneous-Cooperative-Competitive-Single-Policy)
We explored training agents against a copy of themselves (self-play). By adjusting the reward function to favor surviving (longer episode lengths), the agents learned a cooperative strategy of passing the ball to each other to maximize their rewards instead of aggressively trying to score.

## Conclusions

- **Curriculum Learning** proved to be highly effective, yielding a more robust policy in less time compared to standard PPO training.
- **Algorithm Choice Matters:** PPO proved far more suitable for this environment than DQN, likely due to continuous observation spaces and the challenges of off-policy value approximation in this context.
- **Imitation Learning** provided an extremely fast way to achieve a competent agent, albeit capped at the expert's performance.
- **MARL** experiments successfully demonstrated how reward tuning can shift agent behavior from strictly competitive to cooperative.
