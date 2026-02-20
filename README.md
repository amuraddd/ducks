## Diffusion for Offline Reinforcement Learning

### main idea
The main idea is that learning happens sequentially.
We fist master a task at a lower level before mastering the medium level and then finally becoming experts at it.
The data is observed in real life in exactly that way. The tasks are performed at a lower level of efficiency before high efficiency is achieved.

#### Setup
1. Pull offline datasets from Minari
2. Create 4 separate batches of data (use Quartiles to do this: 25%, 50%, 75%, 100%)
    - Low reward batch 0-25
    - Medium reward batch 25-50
    - Meidum-high reward batch 50-75
    - high reward batch 75-100
Try different percentiles and generate different experimental results.
Instead of training separate models - maintain copies of the same inital model. A separate copy for each percentile of data.
Merge all copies at the end through an averaging scheme based on mutual information.
3. Train 4 separate diffusion models. One on each batch.
4. Interpolate all diffusion models by averaging the parameters.
5. Use the final diffusion model to generate new offline episodic data.
6. Learn an agent policy from the generated data using an offline RL algorithm (SAC or DDPG).

### Ablation
To show that training a single diffusion model on the entire training dataset is not as good; the following ablation would need to be done:
- Train a single diffusion model on the entire dataset for each task.
- Measure its performance in comparison to the new proposed method.

### Things to try:
- Train a separate model for each percentile and combine them. Each model with its own loss function.
- Train all models together using a single loss function.

## Part 1
Diffusion model for data generation:
S|A
[s1, s2,....,sn]
[a1, a2,....,an]

Use the idea from AdaptDiffuser to condition the model on rewards:
[r1, r2,....,rn]

## Part 2
- Use the diffusion and reward model from Part 1 to generate new data.
- Train an agent on the newly generated offline dataset.
- Interact with the environment and "hope for better rewards".

#### Results and Comparison
1. Compare the results of the agent policy with offline RL algorithms (SAC, HER, IQL, BC, Diffuser)
    - D4RL
    - Mujoco
2. Find a new unique evaluation criteria and compare performance between the trained agent policy, IQL, and BC.

### Useful papers:
- https://openreview.net/pdf?id=UvQOcw2oCD
- https://arxiv.org/pdf/2205.09991

### Papers which used Diffusion models for RL
- Diffuser: This paper generated whole trajectories: https://arxiv.org/abs/2205.09991
- Decision Diffuser: This paper introduced conditional diffusion with rewards or contraint guidance: https://arxiv.org/pdf/2211.15657
- Diffusion-QL: Adds a regularization term to the loss function of the conditional diffusion model guiding it to learn optimal actions: https://arxiv.org/pdf/2208.06193
- AdaptDiffuser: Trains conditioned diffusion models to be able to generalize on unseen tasks. This paper also filters out trajectories in the synthetic data before retraining on it: https://arxiv.org/pdf/2302.01877