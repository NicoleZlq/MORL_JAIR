import mo_gymnasium as mo_gym
import numpy as np

from morl_baselines.multi_policy.pareto_q_learning.pql import PQL
from mo_gymnasium.utils import MORecordEpisodeStatistics

if __name__ == "__main__":
    env = mo_gym.make("deep-sea-treasure-concave-v0")
    env = MORecordEpisodeStatistics(env, gamma=0.99)
    ref_point = np.array([0, -50])

    agent = PQL(
        env,
        ref_point,
        gamma=0.99,
        initial_epsilon=1.0,
        epsilon_decay_steps=10000,
        final_epsilon=0.1,
        seed=2,
        project_name="MORL-Baselines",
        experiment_name="Concave_DST_PQL",
        log=True,
    )

    pf = agent.train(
        total_timesteps=50000,
        log_every=2000,
        action_eval="hypervolume",
        known_pareto_front=env.pareto_front(gamma=0.99),
        ref_point=ref_point,
        eval_env=env,
    )
    print(pf)

    # Execute a policy
    target = np.array(pf.pop())
    print(f"Tracking {target}")
    reward = agent.track_policy(target, env=env)
    print(f"Obtained {reward}")
