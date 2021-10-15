import gym
import itertools
import numpy as np
import tensorflow.compat.v1 as tf
from tf_slim.layers import layers
import baselines.common.tf_util as U
from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule


class PolicyTrainer:
    def __init__(self, env, num_cpu=8):
        # self.env = env
        self.env = gym.make("CartPole-v0")
        self.num_cpu = num_cpu

    def DQN(self):
        def model(inpt, num_actions, scope, reuse=False):
            """This model takes as input an observation and returns values of all actions."""
            with tf.variable_scope(scope, reuse=reuse):
                out = inpt
                out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
                out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
                return out

        with U.make_session(num_cpu=self.num_cpu):
            # Create all the functions necessary to train the model
            act, train, update_target, debug = deepq.build_train(
                make_obs_ph=lambda name: ObservationInput(self.env.observation_space, name=name),
                q_func=model,
                num_actions=self.env.action_space.n,
                optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
            )
            # Create the replay buffer
            replay_buffer = ReplayBuffer(50000)
            # Create the schedule for exploration starting from 1 (every action is random) down to
            # 0.02 (98% of actions are selected according to values predicted by the model).
            exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

            # Initialize the parameters and copy them to the target network.
            U.initialize()
            update_target()

            episode_rewards = [0.0]
            obs = self.env.reset()
            for t in itertools.count():
                # Take action and update exploration to the newest value
                action = act(obs[None], update_eps=exploration.value(t))[0]
                new_obs, rew, done, _ = self.env.step(action)
                # Store transition in the replay buffer.
                replay_buffer.add(obs, action, rew, new_obs, float(done))
                obs = new_obs

                episode_rewards[-1] += rew
                if done:
                    obs = self.env.reset()
                    episode_rewards.append(0)

                is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
                if is_solved:
                    # Show off the result
                    self.env.render()
                else:
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    if t > 1000:
                        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                        train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                    # Update target network periodically.
                    if t % 1000 == 0:
                        update_target()

                if done and len(episode_rewards) % 10 == 0:
                    logger.record_tabular("steps", t)
                    logger.record_tabular("episodes", len(episode_rewards))
                    logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                    logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                    logger.dump_tabular()

if __name__=="__main__":
    policy_trainer = PolicyTrainer("yxz")
    policy_trainer.DQN()