import fire
import mo_gymnasium as mo_gym
import numpy as np

from morl_baselines.multi_policy.pgmorl.pgmorl import PGMORL



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
 #   eval_env = make_env()  # RecordVideo(make_env(), "videos/minecart/", episode_trigger=lambda e: e % 1000 == 0)

    agent = PGMORL(
        env=env,
        env_id = "mo-hopper-v4",
        origin = 2,
        num_envs=2,
        pop_size=6,
        warmup_iterations=3,
        steps_per_iteration = 6000,
        evolutionary_iterations=4,
        num_weight_candidates=20,
        gamma=0.99,
        log=True,
    )

    agent.train(
        total_timesteps=int(120000),
        eval_env=eval_env,
        ref_point=np.array([-100.0, -100.0]),
        known_pareto_front=None,
    )


if __name__ == "__main__":
    fire.Fire(main)
