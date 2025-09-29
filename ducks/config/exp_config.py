import os
from pathlib import Path

SEEDS = [5, 10, 20, 30]
TRAINING_ARTIFACTS_DIR = "generated_artifacts"

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

def get_mujoco_exp_names_and_replay_paths(baseline="cql", env="halfcheetah", difficulty="random", create_dir=True):
    """
    Create directories for experiments and return dir names and exp names where data will be stored.
    """
    experiments = [
        f"{env}_{difficulty}_{baseline}_seed{seed}"
        # for env in ["halfcheetah", "hopper", "walker2d"]
        # for difficulty in ["random", "medium", "medium_replay", "medium_expert", "expert"]
        for seed in SEEDS
    ]
    
    exp_names = [
        Path(TRAINING_ARTIFACTS_DIR)/experiment
        for experiment in experiments
    ]
    
    replay_path = Path(TRAINING_ARTIFACTS_DIR)/"video"/"mujoco"/baseline/env/difficulty
    # for env in ["halfcheetah", "hopper", "walker2d"]
        
    if create_dir:
        os.chdir(get_project_root()) #set to the root project directory
        # for path in replay_paths:
        if os.path.exists(replay_path):
            print(f"Path {replay_path} exists. Skipping new directory creation.")
        else:
            replay_path.mkdir(parents=True) #this is a pathlike object - when parents are set to True then intermmediate directories are created.
    return exp_names, replay_path
        
        

# EXP_CONFIG = {
#     "CQL": {
#         "exp_names": [
#             Path(TRAINING_ARTIFACTS_DIR)/env+f"_{difficulty}"+f"_cql_seed{seed}"
#             for env in ["halfcheetah", "hopper", "walker2d"]
#             for difficulty in ["random", "medium", "medium_replay", "medium_expert", "expert"]
#             for seed in SEEDS
#         ],
#         "replay_path": [
#             Path(TRAINING_ARTIFACTS_DIR)/"video"/"mujoco"/env
#             for env in ["halfcheetah", "hopper", "walker2d"]
#         ]
#     }
# }


# main_config.exp_name = "generated_artifacts/halfcheetah_expert_cql_seed0"
# main_config.env.replay_path="generated_artifacts/video/mujoco/halfcheetah" #save the video in this location