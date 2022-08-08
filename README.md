Minimal [Decision Transformer](https://github.com/kzl/decision-transformer) Implementation written in Jax (Flax). [[Reference (minimal torch implementation)]](https://github.com/nikhilbarhate99/min-decision-transformer)

## Setup
Set up the environments:
```bash
pip install -r requirements.txt
pip install -e .
```

## Usage
Example:
```bash
# 20k training
CUDA_VISIBLE_DEVICES=0 python scripts/train_dt.py --env halfcheetah --dataset medium
CUDA_VISIBLE_DEVICES=0 python scripts/train_dt.py --env walker2d --dataset medium
CUDA_VISIBLE_DEVICES=0 python scripts/train_dt.py --env hopper --dataset medium

# 100k training
CUDA_VISIBLE_DEVICES=0 python scripts/train_dt.py --env halfcheetah --dataset medium --max_train_iters 20 --policy_save_iters 2 --num_updates_per_iter 5000
CUDA_VISIBLE_DEVICES=0 python scripts/train_dt.py --env walker2d --dataset medium --max_train_iters 20 --policy_save_iters 2 --num_updates_per_iter 5000
CUDA_VISIBLE_DEVICES=0 python scripts/train_dt.py --env hopper --dataset medium --max_train_iters 20 --policy_save_iters 2 --num_updates_per_iter 5000
```

Test example:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/test_dt.py --env halfcheetah --dataset medium --chk_pt_name dt_halfcheetah-medium-v2/seed_0/22-06-22-06-31-49/model_best.pt
```

Citation:
```
@inproceedings{furuta2021generalized,
  title={Generalized Decision Transformer for Offline Hindsight Information Matching},
  author={Hiroki Furuta and Yutaka Matsuo and Shixiang Shane Gu},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```
