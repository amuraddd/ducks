import warnings
warnings.simplefilter('ignore')

# from ducks.utils.utils import set_ld_library_path
# set_ld_library_path()

# from ding.entry import serial_pipeline_offline
from ding.config import compile_config
from dizoo.d4rl.config.halfcheetah_random_cql_config import main_config, create_config

from ding.data import DequeBuffer
from ding.policy import CQLPolicy
from ding.model import ContinuousQAC
from dizoo.d4rl.envs.d4rl_env import D4RLEnv
from ding.envs import DingEnvWrapper, BaseEnvManagerV2
# set_ld_library_path()

from ding.utils import set_pkg_seed
from ding.framework import task, ding_init
from ding.data import create_dataset
from ding.framework.context import OfflineRLContext
from ding.framework.middleware import interaction_evaluator, trainer, CkptSaver, offline_data_fetcher, offline_logger

if __name__=="__main__":
    # set_ld_library_path()
    main_config.exp_name = "generated_artifacts/halfcheetah_expert_cql_seed0"
    main_config.env.replay_path="generated_artifacts/video/mujoco/halfcheetah" #save the video in this location

    cfg = compile_config(
        main_config,
        create_cfg=create_config,
        auto=True,
    )

    model = ContinuousQAC(**cfg.policy.model)
    buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
    policy = CQLPolicy(cfg=cfg.policy, model=model)

    ding_init(cfg)
    with task.start(async_mode=False, ctx=OfflineRLContext()):
        # Evaluating, we place it on the first place to get the score of the random model as a benchmark value
        evaluator_env = BaseEnvManagerV2(
            env_fn=[lambda: D4RLEnv(cfg.env) for _ in range(cfg.env.evaluator_env_num)], cfg=cfg.env.manager
        )
        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
        
        dataset = create_dataset(cfg)
        model = ContinuousQAC(**cfg.policy.model)
        policy = CQLPolicy(cfg.policy, model=model)
        
        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(offline_data_fetcher(cfg, dataset))
        task.use(trainer(cfg, policy.learn_mode))
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=100))
        task.use(offline_logger())
        task.run()