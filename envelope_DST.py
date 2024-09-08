import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.utils import MORecordEpisodeStatistics

from morl_baselines.multi_policy.envelope.envelope import Envelope


def main():
    def make_env():
        env = mo_gym.make("deep-sea-treasure-concave-v0")
        env = MORecordEpisodeStatistics(env, gamma=0.99)
        # env = mo_gym.LinearReward(env)
        return env

    env = make_env()
    eval_env = make_env()
    # RecordVideo(make_env(), "videos/minecart/", episode_trigger=lambda e: e % 1000 == 0)

    agent = Envelope(
        env,
        max_grad_norm=0.1,
        learning_rate=1e-4,
        gamma=0.99,
        batch_size=8,
        net_arch=[256, 256, 256, 256],
        buffer_size=int(2e5),
        initial_epsilon=1.0,
        final_epsilon=0.01,
        epsilon_decay_steps=10000,
        initial_homotopy_lambda=0.9,
        final_homotopy_lambda=0.8,
        homotopy_decay_steps=10000,
        learning_starts=1000,
        envelope=True,  #morl/e
        gradient_updates=1,
        target_net_update_freq=1000,  # 1000,  # 500 reduce by gradient updates
        tau=1,
        log=True,
        project_name="MORL-Baselines",
        experiment_name="IJCAI_Envelope",
        seed=6,
    )

    agent.train(
        total_timesteps=50000,
        total_episodes=None,
        weight=None,
        eval_env=eval_env,
        ref_point=np.array([0,  -50.0]),
        known_pareto_front=env.unwrapped.pareto_front(gamma=0.99),
        num_eval_weights_for_front=5000,
        eval_freq=2000,
        reset_num_timesteps=False,
        reset_learning_starts=False,
    )


if __name__ == "__main__":
    main()
    

