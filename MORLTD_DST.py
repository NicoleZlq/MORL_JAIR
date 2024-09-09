import fire
import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.utils import MORecordEpisodeStatistics
from morl_baselines.multi_policy.td.td import morl_td
import os
import wandb

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
os.chdir(root_dir)


# from gymnasium.wrappers.record_video import RecordVideo


def main( ):
    

    env = MORecordEpisodeStatistics(mo_gym.make("deep-sea-treasure-concave-v0"), gamma=0.99)
    eval_env = mo_gym.make("deep-sea-treasure-concave-v0")


    agent = morl_td(
        env,
        num_nets=2,
        max_grad_norm=None,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=16,
        net_arch=[256, 256, 256],
        buffer_size=int(2e5),
        initial_epsilon=1.0,
        final_epsilon=0.01,
        epsilon_decay_steps=10000,
        learning_starts=1000,
        alpha_per=0.6,
        min_priority=0.001,
        per=True,  # morl/td, gpi/pd
        gpi_pd=True, # morl/td, gpi/pd
        use_gpi=True, # morl/td, gpi/pd, gpi/ls
        gradient_updates=1,
        target_net_update_freq=1000,
        tau=1,
        dyna=False, #gpi/pd
        dynamics_uncertainty_threshold=1.5,
        dynamics_net_arch=[256, 256, 256],
        dynamics_buffer_size=int(1e5),
        dynamics_rollout_batch_size=25000,
        dynamics_train_freq=lambda t: 250,
        dynamics_rollout_freq=1000,
        dynamics_rollout_starts=5000,
        dynamics_rollout_len=5,
        real_ratio=0.5,
        log=True,
        seed= 5,
        project_name="MORL-Baselines",
        experiment_name="Concave_DST_GPI_PD",
        ideal_point = np.array([125, 125]),
        use_td =False,  # morl/td
    )

    agent.train(
        total_timesteps= int(1000 * 50),
        eval_env=eval_env,
        ref_point=np.array([0,-50]),
        known_pareto_front=env.unwrapped.pareto_front(gamma=0.99),
        weight_selection_algo='gpi-ls',
        timesteps_per_iter=int(2000),
        num_eval_weights_for_front= 5000,
    )


if __name__ == "__main__":

    fire.Fire(main)