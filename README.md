## Diffusion for Offline Reinforcement Learning


#### Setup
1. Pull offline datasets from Minari
2. Create 3 separate batches of data
    - Low reward batch
    - Meidum reward batch
    - high reward batch
3. Train 3 separate diffusion models. One on each batch.
4. Interpolate all diffusion models by averaging the parameters.
5. Use the final diffusion model to generate new offline episodic data.
6. Learn an agent policy from the generated data using an offline RL algorithm (SAC or DDPG).

#### Results and Comparison
1. Compare the results of the agent policy with offline RL algorithms (SAC, HER, IQL, BC, Diffuser)
    - D4RL
    - Mujoco
2. Find a new unique evaluation criteria and compare performance between the trained agent policy, IQL, and BC.