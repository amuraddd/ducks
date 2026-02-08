import minari
import torch
import numpy as np
from minari import DataCollector
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
# from gymnasium import spaces


# from ducks.logging.base_logger import logger
from ducks.utils.datat_utils import collate_fn

def load_minari_dataset(dataset_name = "mujoco/hopper/simple-v0"):
    """load dataset"""
    minari_dataset = minari.load_dataset(dataset_name)
    print("Dataset Name: ", dataset_name)
    print("Observation space:", minari_dataset.observation_space)
    print("Action space:", minari_dataset.action_space)
    print("Total episodes:", minari_dataset.total_episodes)
    print("Total steps:", minari_dataset.total_steps)
    return minari_dataset

def get_filtered_episodes_from_minari_dataset(minari_dataset):
    """sample episodes using iterquartile range based on the rewards"""
    episodes = minari_dataset.sample_episodes(n_episodes=minari_dataset.total_episodes)
    # get id's from the sampled episodes
    rewards = list(map(lambda ep: ep.rewards, episodes))
    rewards = np.concatenate(rewards)

    # get quantiles based on episodes not steps
    # combined rewards from all episodes
    # create quantiles from the combined rewards
    # filter episodes by the average reward for each episode being less than a given quantile
    quantiles = [0.25, 0.5, 0.75]
    reward_quantiles = [np.quantile(rewards, quantile) for quantile in quantiles]
    q1_episodes = minari_dataset.filter_episodes(lambda episode: episode.rewards.mean() < reward_quantiles[0])
    q2_episodes = minari_dataset.filter_episodes(lambda episode: episode.rewards.mean() > reward_quantiles[0] and episode.rewards.mean() < reward_quantiles[1])
    q3_episodes = minari_dataset.filter_episodes(lambda episode: episode.rewards.mean() > reward_quantiles[1] and episode.rewards.mean() < reward_quantiles[2])
    q4_episodes = minari_dataset.filter_episodes(lambda episode: episode.rewards.mean() > reward_quantiles[2])

    filtered_episode_datasets = [q1_episodes, q2_episodes, q3_episodes, q4_episodes]
    return filtered_episode_datasets

def get_rewards_by_episode_quantile(filtered_episode_datasets):
    """pull rewards from each filtered episode dataset and return as a list of arrays
    mostly used for plotting reward distributions by episode quantile
    """
    episode_rewards_by_quantile = []
    for filtered_episode_dataset in filtered_episode_datasets:
        if filtered_episode_dataset.total_episodes: #check that there are episodes in the current quantile
            episode_rewards_by_quantile.append(np.concatenate(list(map(lambda ep: ep.rewards, filtered_episode_dataset))))
    return episode_rewards_by_quantile

def get_obs_by_episode_quantile(filtered_episode_datasets):
    """pull observations from each filtered episode dataset and return as a list of arrays
    mostly used for plotting observation distributions by episode quantile
    """
    episode_obs_by_quantile = []
    for filtered_episode_dataset in filtered_episode_datasets:
        if filtered_episode_dataset.total_episodes: #check that there are episodes in the current quantile
            episode_obs_by_quantile.append(np.concatenate(list(map(lambda ep: ep.observations, filtered_episode_dataset))))
    return episode_obs_by_quantile

def get_actions_by_episode_quantile(filtered_episode_datasets):
    """pull actions from each filtered episode dataset and return as a list of arrays
    mostly used for plotting action distributions by episode quantile
    """
    episode_actions_by_quantile = []
    for filtered_episode_dataset in filtered_episode_datasets:
        if filtered_episode_dataset.total_episodes: #check that there are episodes in the current quantile
            episode_actions_by_quantile.append(np.concatenate(list(map(lambda ep: ep.actions, filtered_episode_dataset))))
    return episode_actions_by_quantile


def get_dataloader_from_episodes(filtered_episode_datasets, batch_size=32):
    """
    Get dataloaders from filtered episode datasets
    returns: list of dataloaders. Each dataloader contains episodes for a given quantile.
    """
    dataloaders = []
    for filtered_episode_dataset in filtered_episode_datasets:
        if filtered_episode_dataset.total_episodes:
            # filtered_episode_dataset.set_transform(transform)
            dataloaders.append(
                DataLoader(
                    filtered_episode_dataset, 
                    batch_size=batch_size, 
                    shuffle=True, 
                    collate_fn=collate_fn,
                    num_workers=1
                )
            )
    return dataloaders