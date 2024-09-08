import fire
import mo_gymnasium as mo_gym
import numpy as np

from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import (
    GPIPDContinuousAction,
)


# from gymnasium.wrappers.record_video import RecordVideo


#def main(algo: str, gpi_pd: bool, g: int, timesteps_per_iter: int= 2000):
def main():
    def make_env(record_episode_statistics: bool = False):
        env = mo_gym.make("mo-hopper-v4", cost_objective=False, max_episode_steps=500)
        if record_episode_statistics:
            env = mo_gym.MORecordEpisodeStatistics(env, gamma=0.99)
        return env

    env = make_env(record_episode_statistics=True)
    eval_env = make_env()  

    agent = GPIPDContinuousAction(
        env,
        gradient_updates= 1000,
        min_priority=0.1,
        batch_size=128,
        buffer_size=int(4e5),
        dynamics_rollout_starts=1000,
        dynamics_rollout_len=5,
        dynamics_rollout_freq=250,
        dynamics_rollout_batch_size=50000,
        dynamics_train_freq=1000,
        dynamics_buffer_size=200000,
        dynamics_real_ratio=0.1,
        dynamics_min_uncertainty=2.0,
        dyna=True,  # morl/td(dyna), gpi/pd
        per=True,    # morl/td(dyna), gpi/pd
        project_name="MORL-Baselines",
        experiment_name="MORL/TD",
        log=True,
        seed=1,
        use_td = True, # morl/td(dyna)
        gamma= 0.99,
    )

    agent.train(
        total_timesteps=10 * 6000,
        eval_env=eval_env,
        ref_point=np.array([-100.0, -100.0]),
        known_pareto_front=None,
        weight_selection_algo='gpi-ls',
        timesteps_per_iter=6000,
        ideal_point = np.array([1800,1800]),
    )


if __name__ == "__main__":
    fire.Fire(main)
