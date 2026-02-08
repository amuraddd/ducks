
import seaborn as sns
import matplotlib.pyplot as plt

def plot_reward_distribution_by_episode_quantile(episode_rewards_by_quantile): 
    """plot reward distribution for each reward quantile"""  
    from matplotlib import rcParams
    rcParams['figure.figsize'] = [4, 4]
    
    for i, episode_rewards in enumerate(episode_rewards_by_quantile):
        sns.histplot(episode_rewards, kde=True, alpha=0.5, label=f"Q{i+1} Episode Rewards")
    # Adding labels and legend
    plt.xlabel('Rewards')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
def plot_obs_distribution_by_episode_quantile(episode_obs_by_quantile, mean_along_axis=1): 
    """plot reward distribution for each reward quantile"""  
    from matplotlib import rcParams
    rcParams['figure.figsize'] = [4, 4]
    
    for i, episode_obs in enumerate(episode_obs_by_quantile):
        sns.histplot(episode_obs.mean(axis=mean_along_axis), kde=True, alpha=0.5, label=f"Q{i+1} Episode Observations/States")
    # Adding labels and legend
    plt.xlabel('Observations')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
def plot_action_distribution_by_episode_quantile(episode_action_by_quantile, mean_along_axis=1): 
    """plot reward distribution for each reward quantile"""  
    from matplotlib import rcParams
    rcParams['figure.figsize'] = [4, 4]
    
    for i, episode_action in enumerate(episode_action_by_quantile):
        sns.histplot(episode_action.mean(axis=mean_along_axis), kde=True, alpha=0.5, label=f"Q{i+1} Episode Actions")
    # Adding labels and legend
    plt.xlabel('Actions')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
def plot_dist_from_sample(sample):
    sns.displot(sample.cpu().numpy(), kind="hist", height=2, aspect=1.5)
    plt.show()