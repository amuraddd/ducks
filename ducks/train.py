from ducks.baselines import cql_train
from ducks.logging.base_logger import logger
from ducks.config.exp_config import get_mujoco_exp_names_and_replay_paths

if __name__=="__main__":
    baselines = ["cql"]
    envs = ["halfcheetah", "hopper", "walker2d"]
    difficulty_levels = ["random", "medium", "medium_replay", "medium_expert", "expert"]
    for baseline in baselines:
        for env in envs:
            for difficulty in difficulty_levels:
                exp_names, replay_path = get_mujoco_exp_names_and_replay_paths(
                    baseline=baseline,
                    env=env,
                    difficulty=difficulty,
                    create_dir=True
                )
                for exp_name in exp_names:
                    logger.info(f"Running Baseline: {baseline} | Experiment: {exp_name} | Environment: {env} | Difficulty: {difficulty}")
                    cql_train.run_cql_train(
                        exp_name=exp_name,
                        replay_path=replay_path,
                        train_feq=10000
                    )
                    logger.info(f"Complete Baseline: {baseline} | Experiment: {exp_name} | Environment: {env} | Difficulty: {difficulty}")
        