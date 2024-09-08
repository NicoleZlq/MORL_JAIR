import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium import MORecordEpisodeStatistics

from morl_baselines.common.scalarization import tchebicheff
from morl_baselines.multi_policy.multi_policy_moqlearning.mp_mo_q_learning import (
    MPMOQLearning,
)
if __name__ == "__main__":


    def make_env():
        env = mo_gym.make("deep-sea-treasure-concave-v0")
        env = MORecordEpisodeStatistics(env, gamma=0.99)
        # env = mo_gym.LinearReward(env)
        return env

    env = make_env()
    eval_env = make_env()
    scalarization = tchebicheff(tau=4.0, reward_dim=2)
  #  scalarization = weighted_sum


    mp_moql = MPMOQLearning(
        env,
        learning_rate=0.5,
        scalarization=scalarization,
        use_gpi_policy=False,
        dyna=False,
        initial_epsilon=1,
        final_epsilon=0.1,
        gamma= 0.99,
        epsilon_decay_steps=int(1e5),
        weight_selection_algo="gpi-ls",
        epsilon_ols=0.0,
        seed =6,
        experiment_name = "MOQ-Learning"  ,
        log = True,
        
    )
    mp_moql.train(
        total_timesteps=10 * int(5e3),
        timesteps_per_iteration=int(2e3),
        eval_freq=int(2000),
        eval_env=eval_env,
        ref_point=np.array([0,-50]),
        known_pareto_front=env.unwrapped.pareto_front(gamma=0.99),
    )
