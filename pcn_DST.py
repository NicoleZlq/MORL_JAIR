import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.utils import MORecordEpisodeStatistics

from morl_baselines.multi_policy.pcn.pcn import PCN


def main():
    def make_env():
        env = mo_gym.make("deep-sea-treasure-concave-v0")
        env = MORecordEpisodeStatistics(env, gamma=0.99)
        return env

    env = make_env()

    agent = PCN(
        env,
        scaling_factor=np.array([1, 0.2,0.2]),
        learning_rate=3e-4,
        batch_size=16,
        project_name="MORL-Baselines",
        experiment_name="Concavt_DST_PCN",
        log=True,
        seed=4,
        gamma=0.99
    )

    agent.train(
        eval_env=make_env(),
        total_timesteps=int(51000),
        ref_point=np.array([0,  -50]),
        num_er_episodes=20,
        max_buffer_size=50,
        num_model_updates=50,
        max_return=np.array([125,0]),
        known_pareto_front=env.unwrapped.pareto_front(gamma=0.99),
        timesteps_per_iter=int(2000),
        num_points_pf = 5000,
    )


if __name__ == "__main__":
    main()
