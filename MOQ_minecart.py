import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.utils import MORecordEpisodeStatistics

from morl_baselines.multi_policy.multi_policy_moqlearning.mp_mo_q_learning import MPMOQLearning

def main():
    def make_env():
        env = mo_gym.make("minecart-v0")
        env = MORecordEpisodeStatistics(env, gamma=0.98)
        # env = mo_gym.LinearReward(env)
        return env

    env = make_env()
    eval_env = make_env()
    # RecordVideo(make_env(), "videos/minecart/", episode_trigger=lambda e: e % 1000 == 0)

    agent = MPMOQLearning(
        env,
        learning_rate=3e-4,
        gamma=0.98,
        initial_epsilon=1.0,
        final_epsilon=0.05,
        log=True,
        project_name="MORL-Baselines",
        experiment_name="MultiPolicy MO Q-Learning",
    )

    agent.train(
        total_timesteps=100000,
        eval_env=eval_env,
        ref_point=np.array([0, 0, -200.0]),
        known_pareto_front=env.unwrapped.pareto_front(gamma=0.98),
        num_eval_weights_for_front=100,
        eval_freq=1000
    )


if __name__ == "__main__":
    main()
