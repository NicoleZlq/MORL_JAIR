import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.utils import MORecordEpisodeStatistics

from morl_baselines.multi_policy.envelope.envelope import Envelope


def main():
    def make_env():
        env = mo_gym.make("minecart-v0")
        env = MORecordEpisodeStatistics(env, gamma=0.99)
        # env = mo_gym.LinearReward(env)
        return env

    env = make_env()
    eval_env = make_env()
    # RecordVideo(make_env(), "videos/minecart/", episode_trigger=lambda e: e % 1000 == 0)

    agent = Envelope(
        env,
        max_grad_norm=0.1,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=16,
        net_arch=[256, 256, 256, 256],
        buffer_size=int(2e6),
        initial_epsilon=1.0,
        final_epsilon=0.01,
        epsilon_decay_steps=1e5,
        initial_homotopy_lambda=0.0,
        final_homotopy_lambda=1.0,
        homotopy_decay_steps=10000,
        learning_starts=100,
        envelope=True,  #morl/e
        gradient_updates=10,
        target_net_update_freq=1000,  # 1000,  # 500 reduce by gradient updates
        tau=1,
        log=True,
        project_name="MORL-Baselines",
        experiment_name="IJCAI_MORL_E",
        seed = 3,
    )

    agent.train(
        total_timesteps=100000,
        total_episodes=None,
        weight=None,
        eval_env=eval_env,
        ref_point=np.array([-1, -1, -200.0]),
        known_pareto_front=env.unwrapped.pareto_front(gamma=0.99),
        num_eval_weights_for_front=200,
        eval_freq=2000,
        reset_num_timesteps=False,
        reset_learning_starts=False,
    )



if __name__ == "__main__":
    main()