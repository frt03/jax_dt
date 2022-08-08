import argparse
import gym
import jax
import os
import random

import numpy as np

from decision_transformer.dt.model import make_policy_networks
from decision_transformer.dt.utils import evaluate_on_env, get_d4rl_dataset_stats, get_d4rl_normalized_score, load_params
from decision_transformer.pmap import bcast_local_devices


def test(args):

    eval_dataset = args.dataset         # medium / medium-replay / medium-expert
    eval_rtg_scale = args.rtg_scale     # normalize returns to go

    if args.env == 'walker2d':
        eval_env_name = 'Walker2d-v3'
        eval_rtg_target = args.rtg_target if args.rtg_target is not None else 5000
        eval_env_d4rl_name = f'walker2d-{eval_dataset}-v2'

    elif args.env == 'halfcheetah':
        eval_env_name = 'HalfCheetah-v3'
        eval_rtg_target = args.rtg_target if args.rtg_target is not None else 6000
        eval_env_d4rl_name = f'halfcheetah-{eval_dataset}-v2'

    elif args.env == 'hopper':
        eval_env_name = 'Hopper-v3'
        eval_rtg_target = args.rtg_target if args.rtg_target is not None else 3600
        eval_env_d4rl_name = f'hopper-{eval_dataset}-v2'

    else:
        raise NotImplementedError

    eval_env = gym.make(eval_env_name)

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
    # seed for others
    random.seed(seed)
    np.random.seed(seed)
    eval_env.seed(seed)

    render = args.render                # render the env frames

    num_test_eval_ep = args.num_eval_ep         # num of evaluation episodes
    eval_max_eval_ep_len = args.max_eval_ep_len # max len of one episode

    context_len = args.context_len      # K in decision transformer
    n_blocks = args.n_blocks            # num of transformer blocks
    embed_dim = args.embed_dim          # embedding (hidden) dim of transformer
    n_heads = args.n_heads              # num of transformer heads
    dropout_p = args.dropout_p          # dropout probability


    eval_chk_pt_dir = args.chk_pt_dir

    eval_chk_pt_name = args.chk_pt_name
    eval_chk_pt_list = [eval_chk_pt_name]


    ## manually override check point list
    ## passing a list will evaluate on all checkpoints
    ## and output mean and std score

    #eval_chk_pt_list = [
    #     "dt_halfcheetah-medium-v2/seed_0/22-06-21-16-46-21/model_best.pt",
    #     "dt_halfcheetah-medium-v2/seed_0/22-06-22-06-31-49/model_best.pt",
    #]

    env_data_stats = get_d4rl_dataset_stats(eval_env_d4rl_name)
    eval_state_mean = np.array(env_data_stats['state_mean'])
    eval_state_std = np.array(env_data_stats['state_std'])

    state_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]

    all_scores = []

    policy_model = make_policy_networks(
        state_dim=state_dim,
        act_dim=act_dim,
        n_blocks=n_blocks,
        h_dim=embed_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=dropout_p,
    )

    for eval_chk_pt_name, key_i in zip(eval_chk_pt_list, jax.random.split(key, num=len(eval_chk_pt_list))):

        eval_chk_pt_path = os.path.join(eval_chk_pt_dir, eval_chk_pt_name)

        # load checkpoint
        policy_params = load_params(eval_chk_pt_path)
        print("model loaded from: " + eval_chk_pt_path)
        # count the number of parameters
        param_count = sum(x.size for x in jax.tree_leaves(policy_params))
        print(f'num_policy_param: {param_count}')

        policy_params = bcast_local_devices(policy_params, local_devices_to_use)

        # evaluate on env
        results = evaluate_on_env(policy_model,
                                  policy_params,
                                  key_i,
                                  context_len,
                                  eval_env,
                                  eval_rtg_target,
                                  eval_rtg_scale,
                                  num_test_eval_ep,
                                  eval_max_eval_ep_len,
                                  eval_state_mean,
                                  eval_state_std,
                                  render=render)
        print(results)

        norm_score = get_d4rl_normalized_score(results['eval/avg_reward'], eval_env_name) * 100
        print("normalized d4rl score: " + format(norm_score, ".5f"))

        all_scores.append(norm_score)

    print("=" * 60)
    all_scores = np.array(all_scores)
    print("evaluated on env: " + eval_env_name)
    print("total num of checkpoints evaluated: " + str(len(eval_chk_pt_list)))
    print("d4rl score mean: " + format(all_scores.mean(), ".5f"))
    print("d4rl score std: " + format(all_scores.std(), ".5f"))
    print("d4rl score var: " + format(all_scores.var(), ".5f"))
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

    parser.add_argument("--render", action="store_true", default=False)

    parser.add_argument('--chk_pt_dir', type=str, default='dt_runs/')
    parser.add_argument('--chk_pt_name', type=str,
                        default='dt_halfcheetah-medium-v2/seed_0/22-06-22-06-31-49/model_best.pt')

    parser.add_argument('--context_len', type=int, default=20)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--dropout_p', type=float, default=0.1)

    parser.add_argument('--max_devices_per_host', type=int, default=None)

    args = parser.parse_args()

    test(args)
