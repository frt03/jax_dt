import argparse
import csv
import flax
import functools
import gym
import jax
import optax
import os
import pickle
import random
import sys

import jax.numpy as jnp
import numpy as np

from datetime import datetime
from typing import Any, Dict, Tuple

from decision_transformer.dt.model import make_policy_networks
from decision_transformer.dt.utils import ReplayBuffer, TrainingState, Transition
from decision_transformer.dt.utils import discount_cumsum, evaluate_on_env, get_d4rl_normalized_score, save_params
from decision_transformer.pmap import bcast_local_devices, synchronize_hosts, is_replicated

def train(args):

    dataset = args.dataset          # medium / medium-replay / medium-expert
    rtg_scale = args.rtg_scale      # normalize returns to go

    # use v3 env for evaluation because
    # Decision Transformer paper evaluates results on v3 envs

    if args.env == 'walker2d':
        env_name = 'Walker2d-v3'
        rtg_target = args.rtg_target if args.rtg_target is not None else 5000
        env_d4rl_name = f'walker2d-{dataset}-v2'

    elif args.env == 'halfcheetah':
        env_name = 'HalfCheetah-v3'
        rtg_target = args.rtg_target if args.rtg_target is not None else 6000
        env_d4rl_name = f'halfcheetah-{dataset}-v2'

    elif args.env == 'hopper':
        env_name = 'Hopper-v3'
        rtg_target = args.rtg_target if args.rtg_target is not None else 3600
        env_d4rl_name = f'hopper-{dataset}-v2'

    else:
        raise NotImplementedError

    env = gym.make(env_name)

    # device settings
    max_devices_per_host = args.max_devices_per_host
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    print(f'Device count: {jax.device_count()}, process count: {process_count} (id {process_id}), local device count: {local_device_count}, devices to be used count: {local_devices_to_use}')

    # seed for jax
    seed = args.seed
    key = jax.random.PRNGKey(seed)
    global_key, local_key, test_key = jax.random.split(key, 3)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    # seed for others
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    max_eval_ep_len = args.max_eval_ep_len  # max len of one episode
    num_eval_ep = args.num_eval_ep          # num of evaluation episodes

    batch_size = args.batch_size            # training batch size
    batch_size_per_device = batch_size // local_devices_to_use
    grad_updates_per_step = args.grad_updates_per_step

    lr = args.lr                            # learning rate
    wt_decay = args.wt_decay                # weight decay
    warmup_steps = args.warmup_steps        # warmup steps for lr scheduler

    # total updates = max_train_iters x num_updates_per_iter
    max_train_iters = args.max_train_iters
    num_updates_per_iter = args.num_updates_per_iter

    context_len = args.context_len      # K in decision transformer
    n_blocks = args.n_blocks            # num of transformer blocks
    embed_dim = args.embed_dim          # embedding (hidden) dim of transformer
    n_heads = args.n_heads              # num of transformer heads
    dropout_p = args.dropout_p          # dropout probability

    # load data from this file
    dataset_path = f'{args.dataset_dir}/{env_d4rl_name}.pkl'

    # saves model and csv in this directory
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    prefix = "dt_" + env_d4rl_name

    log_dir = os.path.join(log_dir, prefix, f'seed_{seed}', start_time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    save_model_name = "model.pt"
    save_model_path = os.path.join(log_dir, save_model_name)
    save_best_model_path = save_model_path[:-3] + "_best.pt"

    log_csv_name = "log.csv"
    log_csv_path = os.path.join(log_dir, log_csv_name)

    csv_writer = csv.writer(open(log_csv_path, 'a', 1))
    csv_header = ([
        "duration",
        "num_updates",
        "action_loss",
        "eval_avg_reward",
        "eval_avg_ep_len",
        "eval_d4rl_score"
    ])

    csv_writer.writerow(csv_header)

    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)

    print("dataset path: " + dataset_path)
    print("model save path: " + save_model_path)
    print("log csv save path: " + log_csv_path)

    # load dataset
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    trans_dim = state_dim + act_dim + 1 + 1 + 1  # rtg, timesteps, mask

    # to get status
    max_epi_len = -1
    min_epi_len = 10**6
    state_stats = []
    for traj in trajectories:
        traj_len = traj['observations'].shape[0]
        min_epi_len = min(min_epi_len, traj_len)
        max_epi_len = max(max_epi_len, traj_len)
        state_stats.append(traj['observations'])
        # convert
        traj['actions'] = jnp.array(traj['actions'])
        traj['observations'] = jnp.array(traj['observations'])
        # calculate returns to go and rescale them
        traj['returns_to_go'] = jnp.array(discount_cumsum(traj['rewards'], 1.0) / rtg_scale).reshape(-1, 1)
        traj['timesteps'] = jnp.arange(start=0, stop=traj_len, step=1, dtype=jnp.int32).reshape(-1, 1)
        traj['traj_mask'] = jnp.ones(traj_len).reshape(-1, 1)

    # used for input normalization
    state_stats = jnp.concatenate(state_stats, axis=0)
    state_mean, state_std = jnp.mean(state_stats, axis=0), jnp.std(state_stats, axis=0) + 1e-8

    # apply padding
    replay_buffer_data = []
    for traj in trajectories:
        traj_len = traj['observations'].shape[0]
        padding_len = (max_epi_len + context_len) - traj_len
        states = traj['observations']

        # apply input normalization
        if not args.rm_normalization:
            states = (states - state_mean) / state_std

        states = jnp.concatenate([states, jnp.zeros((padding_len, state_dim))], axis=0)
        actions = jnp.concatenate([traj['actions'], jnp.zeros((padding_len, act_dim))], axis=0)
        returns_to_go = jnp.concatenate([traj['returns_to_go'], jnp.zeros((padding_len, 1))], axis=0)
        timesteps = jnp.concatenate([traj['timesteps'], jnp.zeros((padding_len, 1))], axis=0)
        traj_mask = jnp.concatenate([traj['traj_mask'], jnp.zeros((padding_len, 1))], axis=0)

        padding_data = jnp.concatenate([states, actions, returns_to_go, timesteps, traj_mask], axis=-1)
        assert trans_dim == padding_data.shape[-1], padding_data.shape
        replay_buffer_data.append(padding_data)

    replay_buffer = ReplayBuffer(
        data=jnp.concatenate(replay_buffer_data, axis=0).reshape(local_devices_to_use, -1, max_epi_len + context_len, trans_dim)
    ) # (local_devices_to_use, num_epi, max_epi_len + context_len, trans_dim)

    policy_model = make_policy_networks(
        state_dim=state_dim,
        act_dim=act_dim,
        n_blocks=n_blocks,
        h_dim=embed_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=dropout_p,
    )

    schedule_fn = optax.polynomial_schedule(
        init_value=lr * 1 / warmup_steps,
        end_value=lr,
        power=1,
        transition_steps=warmup_steps,
        transition_begin=0
    )
    policy_optimizer = optax.chain(
        optax.clip(args.gradient_clipping),
        optax.adamw(learning_rate=schedule_fn, weight_decay=wt_decay),
    )
    key_params, key_dropout = jax.random.split(global_key)
    policy_params = policy_model.init({'params': key_params, 'dropout': key_dropout})
    policy_optimizer_state = policy_optimizer.init(policy_params)

    # count the number of parameters
    param_count = sum(x.size for x in jax.tree_leaves(policy_params))
    print(f'num_policy_param: {param_count}')

    policy_optimizer_state, policy_params = bcast_local_devices(
        (policy_optimizer_state, policy_params), local_devices_to_use)

    def actor_loss(policy_params: Any,
                   transitions: Transition, key: jnp.ndarray) -> jnp.ndarray:
        ts = transitions.ts.reshape(transitions.ts.shape[:2]).astype(jnp.int32)  # (batch_size_per_device, context_len)
        s_t = transitions.s_t  # (batch_size_per_device, context_len, state_dim)
        a_t = transitions.a_t  # (batch_size_per_device, context_len, action_dim)
        rtg_t = transitions.rtg_t  # (batch_size_per_device, context_len, 1)
        mask = transitions.mask_t  # (batch_size_per_device, context_len, 1)
        _, a_p, _ = policy_model.apply(policy_params, ts, s_t, a_t, rtg_t, rngs={'dropout': key})

        a_t = jnp.where(mask.reshape(-1, 1) > 0, a_t.reshape(-1, act_dim), jnp.zeros(()))
        a_p = jnp.where(mask.reshape(-1, 1) > 0, a_p.reshape(-1, act_dim), jnp.zeros(()))

        actor_loss = jnp.mean(jnp.square(a_t - a_p))

        return actor_loss

    actor_grad = jax.jit(jax.value_and_grad(actor_loss))

    @jax.jit
    def update_step(
        state: TrainingState,
        transitions: jnp.ndarray,
    ) -> Tuple[TrainingState, bool, Dict[str, jnp.ndarray]]:

        transitions = Transition(
            s_t=transitions[:, :, :state_dim],
            a_t=transitions[:, :, state_dim:state_dim+act_dim],
            rtg_t=transitions[:, :, state_dim+act_dim:state_dim+act_dim+1],
            ts=transitions[:, :, state_dim+act_dim+1:state_dim+act_dim+1+1],
            mask_t=transitions[:, :, state_dim+act_dim+1+1:state_dim+act_dim+1+1+1]
        )

        key, key_actor = jax.random.split(state.key, 2)

        actor_loss, actor_grads = actor_grad(state.policy_params, transitions, key_actor)
        actor_grads = jax.lax.pmean(actor_grads, axis_name='i')
        policy_params_update, policy_optimizer_state = policy_optimizer.update(
            actor_grads, state.policy_optimizer_state, state.policy_params)
        policy_params = optax.apply_updates(state.policy_params, policy_params_update)

        metrics = {'actor_loss': actor_loss}

        new_state = TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            key=key,
            actor_steps=state.actor_steps + 1)
        return new_state, metrics

    def sample_data(training_state, replay_buffer):
        # num_updates_per_iter
        key1, key2, key3 = jax.random.split(training_state.key, 3)
        epi_idx = jax.random.randint(
            key1, (int(batch_size_per_device*grad_updates_per_step),),
            minval=0,
            maxval=replay_buffer.data.shape[0])  # from (0, num_epi)
        context_idx = jax.random.randint(
            key2, (int(batch_size_per_device*grad_updates_per_step),),
            minval=0,
            maxval=replay_buffer.data.shape[1])  # from (0, max_epi_len)

        def dynamic_slice_context(carry, x):
            traj, c_idx = x
            return (), jax.lax.dynamic_slice(traj, (c_idx, 0), (context_len, trans_dim))

        # (batch_size_per_device*num_updates_per_iter, max_epi_len + context_len, trans_dim)
        transitions = jnp.take(replay_buffer.data, epi_idx, axis=0, mode='clip')
        # (batch_size_per_device*num_updates_per_iter, context_len, trans_dim)
        _, transitions = jax.lax.scan(dynamic_slice_context, (), (transitions, context_idx))
        # (num_updates_per_iter, batch_size_per_device, context_len, trans_dim)
        transitions = jnp.reshape(transitions,
                                [grad_updates_per_step, -1] + list(transitions.shape[1:]))

        training_state = training_state.replace(key=key3)
        return training_state, transitions

    def run_one_epoch(carry, unused_t):
        training_state, replay_buffer = carry

        training_state, transitions = sample_data(training_state, replay_buffer)
        training_state, metrics = jax.lax.scan(
            update_step, training_state, transitions, length=1)
        return (training_state, replay_buffer), metrics

    def run_training(training_state, replay_buffer):
        synchro = is_replicated(
            training_state.replace(key=jax.random.PRNGKey(0)), axis_name='i')
        (training_state, replay_buffer), metrics = jax.lax.scan(
            run_one_epoch, (training_state, replay_buffer), (),
            length=num_updates_per_iter)
        metrics = jax.tree_map(jnp.mean, metrics)
        return training_state, replay_buffer, metrics, synchro
    
    run_training = jax.pmap(run_training, axis_name='i')

    training_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        key=jnp.stack(jax.random.split(local_key, local_devices_to_use)),
        actor_steps=jnp.zeros((local_devices_to_use,)))

    max_d4rl_score = -1.0
    total_updates = 0

    for i_train_iter in range(max_train_iters):
        log_action_losses = []

        # optimization
        training_state, replay_buffer, training_metrics, synchro = run_training(
            training_state, replay_buffer)
        assert synchro[0], (current_step, training_state)
        jax.tree_map(lambda x: x.block_until_ready(), training_metrics)
        log_action_losses.append(training_metrics['actor_loss'])

        # evaluate action accuracy
        results = evaluate_on_env(policy_model,
                                  training_state.policy_params,
                                  test_key,
                                  context_len,
                                  env,
                                  rtg_target,
                                  rtg_scale,
                                  num_eval_ep,
                                  max_eval_ep_len,
                                  state_mean,
                                  state_std)

        eval_avg_reward = results['eval/avg_reward']
        eval_avg_ep_len = results['eval/avg_ep_len']
        eval_d4rl_score = get_d4rl_normalized_score(results['eval/avg_reward'], env_name) * 100

        mean_action_loss = np.mean(log_action_losses)
        time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)

        total_updates += num_updates_per_iter

        log_str = ("=" * 60 + '\n' +
                   "time elapsed: " + time_elapsed  + '\n' +
                   "train iter: " + str(i_train_iter)  + '\n' +
                   "num of updates: " + str(total_updates) + '\n' +
                   "action loss: " +  format(mean_action_loss, ".5f") + '\n' +
                   "eval avg reward: " + format(eval_avg_reward, ".5f") + '\n' +
                   "eval avg ep len: " + format(eval_avg_ep_len, ".5f") + '\n' +
                   "eval d4rl score: " + format(eval_d4rl_score, ".5f")
                )

        print(log_str)

        log_data = [
            time_elapsed,
            total_updates,
            mean_action_loss,
            eval_avg_reward,
            eval_avg_ep_len,
            eval_d4rl_score
        ]

        csv_writer.writerow(log_data)

        # save model
        _policy_params = jax.tree_map(lambda x: x[0], training_state.policy_params)
        print("max d4rl score: " + format(max_d4rl_score, ".5f"))
        if eval_d4rl_score >= max_d4rl_score:
            print("saving max d4rl score model at: " + save_best_model_path)
            save_params(save_best_model_path, _policy_params)
            max_d4rl_score = eval_d4rl_score

        if i_train_iter % args.policy_save_iters == 0 or i_train_iter == max_train_iters - 1:
            save_current_model_path = save_model_path[:-3] + f"_{total_updates}.pt"
            print("saving current model at: " + save_current_model_path)
            save_params(save_current_model_path, _policy_params)

    synchronize_hosts()

    print("=" * 60)
    print("finished training!")
    print("=" * 60)
    end_time = datetime.now().replace(microsecond=0)
    time_elapsed = str(end_time - start_time)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("started training at: " + start_time_str)
    print("finished training at: " + end_time_str)
    print("total training time: " + time_elapsed)
    print("max d4rl score: " + format(max_d4rl_score, ".5f"))
    print("saved max d4rl score model at: " + save_best_model_path)
    print("saved last updated model at: " + save_model_path)
    print("=" * 60)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env', type=str, default='halfcheetah')
    parser.add_argument('--dataset', type=str, default='medium')
    parser.add_argument('--rtg_scale', type=int, default=1000)
    parser.add_argument('--rtg_target', type=int, default=None)

    parser.add_argument('--max_eval_ep_len', type=int, default=1000)
    parser.add_argument('--num_eval_ep', type=int, default=10)

    parser.add_argument('--dataset_dir', type=str, default='data/')
    parser.add_argument('--log_dir', type=str, default='dt_runs/')

    parser.add_argument('--context_len', type=int, default=20)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--gradient_clipping', type=float, default=0.25)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--grad_updates_per_step', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wt_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)

    parser.add_argument('--max_train_iters', type=int, default=200)
    parser.add_argument('--num_updates_per_iter', type=int, default=100)
    parser.add_argument('--policy_save_iters', type=int, default=10)
    parser.add_argument('--rm_normalization', action='store_true', help='Turn off input normalization')

    parser.add_argument('--max_devices_per_host', type=int, default=None)

    args = parser.parse_args()

    train(args)
