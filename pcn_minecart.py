import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.utils import MORecordEpisodeStatistics

from morl_baselines.multi_policy.pcn.pcn import PCN


def main():
    def make_env():
        env = mo_gym.make("minecart-v0")
        env = MORecordEpisodeStatistics(env, gamma=0.99)
        return env

    env = make_env()

    agent = PCN(
        env,
        scaling_factor=np.array([1, 1, 0.1, 0.1]),
        learning_rate=3e-4,
        batch_size=256,
        project_name="MORL-Baselines",
        experiment_name="PCN",
        log=True,
        gamma= 0.99,
        seed=2,
    )

    agent.train(
        eval_env=make_env(),
        total_timesteps=int(101000),
        ref_point=np.array([-1, -1, -200.0]),
        num_er_episodes=20,
        max_buffer_size=50,
        num_model_updates=50,
        max_return=np.array([1.5,1.5,0]),
        known_pareto_front=env.unwrapped.pareto_front(gamma=0.99),
        timesteps_per_iter=int(2000),
        num_points_pf = 200,
    )


if __name__ == "__main__":
    main()
