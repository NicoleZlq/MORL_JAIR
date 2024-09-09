import fire
import mo_gymnasium as mo_gym
import numpy as np

from morl_baselines.multi_policy.td.td import morl_td


# from gymnasium.wrappers.record_video import RecordVideo


def main( ):
    def make_env():
        env = mo_gym.make("minecart-v0")
        env = mo_gym.MORecordEpisodeStatistics(env, gamma=0.99)
        return env

    env = make_env()
    eval_env = make_env()  # RecordVideo(make_env(), "videos/minecart/", episode_trigger=lambda e: e % 1000 == 0)

    agent = morl_td(
        env,
        num_nets=2,
        max_grad_norm=None,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=128,
        net_arch=[256, 256, 256, 256],
        buffer_size=int(1e6),
        initial_epsilon=1.0,
        final_epsilon=0.01,
        epsilon_decay_steps=1e5,
        learning_starts=100,
        alpha_per=0.6,
        min_priority=0.05,
        per=True, # morl/td, morl/td(dyna), gpi/pd
        gpi_pd=True,  # morl/td, morl/td(dyna), gpi/pd
        use_gpi=True,  # morl/td, morl/td(dyna), gpi/pd, gpi/ls
        gradient_updates=1000,
        target_net_update_freq=200,
        tau=1,
        dyna=True,  # morl/td(dyna), gpi/pd
        dynamics_uncertainty_threshold=1.5,
        dynamics_net_arch=[256, 256, 256],
        dynamics_buffer_size=int(1e5),
        dynamics_rollout_batch_size=25000,
        dynamics_train_freq=lambda t: 250,
        dynamics_rollout_freq=250,
        dynamics_rollout_starts=5000,
        dynamics_rollout_len=1,
        real_ratio=0.5,
        log=True,
        seed =5,
        project_name="MORL-Baselines",
        experiment_name="IJCAI_GPI_PD",
        ideal_point = np.array([2,2, 2]),
        use_td =False,  # morl/td, morl/td(dyna)
    )

    agent.train(
        total_timesteps= 10000 * 10,
        eval_env=eval_env,
        ref_point=np.array([-1, -1, -200.0]),
        known_pareto_front=env.unwrapped.pareto_front(gamma=0.99),
        weight_selection_algo='gpi-ls', #gpi-ls
        timesteps_per_iter=2000,
        num_eval_weights_for_front= 200,
        
    )


if __name__ == "__main__":
    fire.Fire(main)
0