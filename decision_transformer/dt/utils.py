import flax
import jax
import optax
import pickle
import random
import time

import jax.numpy as jnp
import numpy as np

from typing import Any
from decision_transformer.d4rl_infos import REF_MIN_SCORE, REF_MAX_SCORE, D4RL_DATASET_STATS


def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t+1]
    return disc_cumsum


def get_d4rl_normalized_score(score, env_name):
    env_key = env_name.split('-')[0].lower()
    assert env_key in REF_MAX_SCORE, f'no reference score for {env_key} env to calculate d4rl score'
    return (score - REF_MIN_SCORE[env_key]) / (REF_MAX_SCORE[env_key] - REF_MIN_SCORE[env_key])


def get_d4rl_dataset_stats(env_d4rl_name):
    return D4RL_DATASET_STATS[env_d4rl_name]


def evaluate_on_env(policy_model, policy_params, key, context_len, env, rtg_target, rtg_scale,
                    num_eval_ep=10, max_test_ep_len=1000,
                    state_mean=None, state_std=None, render=False):

    policy_params = jax.tree_map(lambda x: x[0], policy_params)
    policy_model_apply = jax.jit(policy_model.apply)

    eval_batch_size = 1  # required for forward pass

    results = {}
    total_reward = 0
    total_timesteps = 0

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    if state_mean is None:
        state_mean = jnp.zeros((state_dim,))
    else:
        state_mean = jnp.array(state_mean)

    if state_std is None:
        state_std = jnp.ones((state_dim,))
    else:
        state_std = jnp.array(state_std)

    # same as timesteps used for training the transformer
    # also, crashes if device is passed to arange()
    timesteps = jnp.arange(start=0, stop=max_test_ep_len, step=1, dtype=jnp.int32)
    timesteps = jnp.tile(timesteps, (eval_batch_size, 1))

    for _, key_i in enumerate(jax.random.split(key, num=num_eval_ep)):
        # zeros place holders
        actions = jnp.zeros((eval_batch_size, max_test_ep_len, act_dim),
                            dtype=jnp.float32)
        states = jnp.zeros((eval_batch_size, max_test_ep_len, state_dim),
                            dtype=jnp.float32)
        rewards_to_go = jnp.zeros((eval_batch_size, max_test_ep_len, 1),
                            dtype=jnp.float32)

        # init episode
        running_state = env.reset()
        running_reward = 0
        running_rtg = rtg_target / rtg_scale

        for t in range(max_test_ep_len):

            total_timesteps += 1

            # add state in placeholder and normalize
            running_state = (jnp.array(running_state) - state_mean) / state_std
            states = jax.ops.index_update(states, jnp.index_exp[0, t], jnp.array(running_state))

            # calcualate running rtg and add it in placeholder
            running_rtg = running_rtg - (running_reward / rtg_scale)
            rewards_to_go = jax.ops.index_update(rewards_to_go, jnp.index_exp[0, t], jnp.array(running_rtg))

            if t < context_len:
                _, act_preds, _ = policy_model_apply(policy_params,
                                                     timesteps[:,:context_len],
                                                     states[:,:context_len],
                                                     actions[:,:context_len],
                                                     rewards_to_go[:,:context_len],
                                                     rngs={'dropout': key_i})
                act = act_preds[0, t]
            else:
                _, act_preds, _ = policy_model_apply(policy_params,
                                                     timesteps[:,t-context_len+1:t+1],
                                                     states[:,t-context_len+1:t+1],
                                                     actions[:,t-context_len+1:t+1],
                                                     rewards_to_go[:,t-context_len+1:t+1],
                                                     rngs={'dropout': key_i})
                act = act_preds[0, -1]

            running_state, running_reward, done, _ = env.step(act)

            # add action in placeholder
            actions = jax.ops.index_update(actions, jnp.index_exp[0, t], act)

            total_reward += running_reward

            if render:
                env.render()
            if done:
                break

    results['eval/avg_reward'] = total_reward / num_eval_ep
    results['eval/avg_ep_len'] = total_timesteps / num_eval_ep

    return results


@flax.struct.dataclass
class ReplayBuffer:
    """Contains data related to a replay buffer."""
    data: jnp.ndarray


@flax.struct.dataclass
class Transition:
    """Contains data for contextual-BC training step."""
    s_t: jnp.ndarray
    a_t: jnp.ndarray
    rtg_t: jnp.ndarray
    ts: jnp.ndarray
    mask_t: jnp.ndarray


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    policy_optimizer_state: optax.OptState
    policy_params: Any
    key: jnp.ndarray
    actor_steps: jnp.ndarray


class File:
    """General purpose file resource."""
    def __init__(self, fileName: str, mode='r'):
        self.f = None
        if not self.f:
            self.f = open(fileName, mode)

    def __enter__(self):
        return self.f

    def __exit__(self, exc_type, exc_value, traceback):
        self.f.close()


def save_params(path: str, params: Any):
    """Saves parameters in Flax format."""
    with File(path, 'wb') as fout:
        fout.write(pickle.dumps(params))


def load_params(path: str) -> Any:
  with File(path, 'rb') as fin:
    buf = fin.read()
  return pickle.loads(buf)
