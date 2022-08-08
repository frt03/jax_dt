"""
From https://github.com/nikhilbarhate99/min-decision-transformer/blob/master/decision_transformer/model.py
Causal transformer (GPT) implementation
"""
import dataclasses
import jax
import jax.numpy as jnp

from flax import linen
from flax.linen.initializers import lecun_normal, zeros
from typing import Any, Callable, Optional


@dataclasses.dataclass
class FeedForwardModel:
    init: Any
    apply: Any


class MaskedCausalAttention(linen.Module):
    h_dim: int
    max_T: int
    n_heads: int
    drop_p: float = 0.1
    dtype: Any = jnp.float32
    kernel_init: Callable[..., Any] = lecun_normal()
    bias_init: Callable[..., Any] = zeros
    deterministic: bool = False if drop_p > 0.0 else True

    def setup(self):
        self.mask = jnp.tril(
            jnp.ones((self.max_T, self.max_T))).reshape(1, 1, self.max_T, self.max_T)

    @linen.compact
    def __call__(self, src: jnp.ndarray) -> jnp.ndarray:
        B, T, C = src.shape # batch size, seq length, h_dim * n_heads
        N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim
        
        # rearrange q, k, v as (B, N, T, D)
        q = linen.Dense(
            self.h_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(src).reshape(B, T, N, D).transpose(0, 2, 1, 3)
        k = linen.Dense(
            self.h_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(src).reshape(B, T, N, D).transpose(0, 2, 1, 3)
        v = linen.Dense(
            self.h_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(src).reshape(B, T, N, D).transpose(0, 2, 1, 3)
        
        # weights (B, N, T, T)
        weights = q @ k.transpose(0, 1, 3, 2) / jnp.sqrt(D)

        # causal mask applied to weights
        # mask == True --> weights, mask == False --> -jnp.inf
        weights = jnp.where(self.mask[..., :T, :T], weights, -jnp.inf)
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = jax.nn.softmax(weights, axis=-1)

        attention = linen.Dropout(
            rate=self.drop_p,
            deterministic=self.deterministic)(normalized_weights @ v)
        
        attention = attention.transpose(0, 2, 1, 3).reshape(B, T, N*D)

        projection = linen.Dense(
            self.h_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(attention)

        out = linen.Dropout(
            rate=self.drop_p,
            deterministic=self.deterministic)(projection)

        return out


class Block(linen.Module):
    h_dim: int
    max_T: int
    n_heads: int
    drop_p: float = 0.1
    dtype: Any = jnp.float32
    kernel_init: Callable[..., Any] = lecun_normal()
    bias_init: Callable[..., Any] = zeros
    deterministic: bool = False if drop_p > 0.0 else True

    @linen.compact
    def __call__(self, src: jnp.ndarray) -> jnp.ndarray:
        # Attention -> LayerNorm -> MLP -> LayerNorm
        src = src + MaskedCausalAttention(
            h_dim=self.h_dim,
            max_T=self.max_T,
            n_heads=self.n_heads,
            drop_p=self.drop_p,
        )(src) # residual
        src = linen.LayerNorm(dtype=self.dtype)(src)

        src2 = linen.Dense(
            self.h_dim*4,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(src)
        src2 = jax.nn.gelu(src2)
        src2 = linen.Dense(
            self.h_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(src2)
        src2 = linen.Dropout(
            rate=self.drop_p,
            deterministic=self.deterministic)(src2)

        src = src + src2 # residual
        src = linen.LayerNorm(dtype=self.dtype)(src)
        return src


class DecisionTransformer(linen.Module):
    state_dim: int
    act_dim: int
    n_blocks: int
    h_dim: int
    context_len: int
    n_heads: int
    drop_p: float
    dtype: Any = jnp.float32
    max_timestep: int = 4096
    use_action_tanh: bool = True
    kernel_init: Callable[..., Any] = lecun_normal()
    bias_init: Callable[..., Any] = zeros

    def setup(self):
        self.input_seq_len = 3 * self.context_len
    
    @linen.compact
    def __call__(self,
                 timesteps: jnp.ndarray,
                 states: jnp.ndarray,
                 actions: jnp.ndarray,
                 returns_to_go: jnp.ndarray) -> jnp.ndarray:
        B, T, _ = states.shape

        time_embeddings = linen.Embed(
            num_embeddings=self.max_timestep,
            features=self.h_dim)(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = linen.Dense(
            self.h_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(states) + time_embeddings
        action_embeddings = linen.Dense(
            self.h_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(actions) + time_embeddings
        returns_embeddings = linen.Dense(
            self.h_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(returns_to_go) + time_embeddings

        # stack rtg, states and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)
        h = jnp.stack(
            (returns_embeddings, state_embeddings, action_embeddings), axis=1
        ).transpose(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)

        h = linen.LayerNorm(dtype=self.dtype)(h)

        # transformer and prediction
        for _ in range(self.n_blocks):
            h = Block(
                h_dim=self.h_dim,
                max_T=self.input_seq_len,
                n_heads=self.n_heads,
                drop_p=self.drop_p)(h)
        
        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t, a_t
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus 
        # the 3 input variables at that timestep (r_t, s_t, a_t) in sequence.
        h = h.reshape(B, T, 3, self.h_dim).transpose(0, 2, 1, 3)

        # get predictions
        return_preds = linen.Dense(
            1,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(h[:, 2])     # predict next rtg given r, s, a
        state_preds = linen.Dense(
            self.state_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(h[:, 2])      # predict next state given r, s, a
        action_preds = linen.Dense(
            self.act_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(h[:, 1])      # predict action given r, s
        if self.use_action_tanh:
            action_preds = jnp.tanh(action_preds)

        return state_preds, action_preds, return_preds


def make_transformer(state_dim: int,
                     act_dim: int,
                     n_blocks: int,
                     h_dim: int,
                     context_len: int,
                     n_heads: int,
                     drop_p: float) -> DecisionTransformer:
    """Creates a DecisionTransformer model.
    Args:
        state_dim: dimension of state
        act_dim: dimension of action
        n_blocks: number of attention blocks in transformer
        h_dim: size of hidden unit for liner layers
        context_len: length of context
        n_heads: number of attention heads in in transformer
        drop_p: dropout rate
    Returns:
        a model
    """
    module = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        n_blocks=n_blocks,
        h_dim=h_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=drop_p)

    return module


def make_policy_networks(state_dim: int,
                         act_dim: int,
                         n_blocks: int,
                         h_dim: int,
                         context_len: int,
                         n_heads: int,
                         drop_p: float) -> FeedForwardModel:
    batch_size = 1
    dummy_timesteps = jnp.zeros((batch_size, context_len), dtype=jnp.int32)
    dummy_states = jnp.zeros((batch_size, context_len, state_dim))
    dummy_actions = jnp.zeros((batch_size, context_len, act_dim))
    dummy_rtg = jnp.zeros((batch_size, context_len, 1))

    def policy_model_fn():
        class PolicyModule(linen.Module):
            @linen.compact
            def __call__(self,
                         timesteps: jnp.ndarray,
                         states: jnp.ndarray,
                         actions: jnp.ndarray,
                         returns_to_go: jnp.ndarray):
                s_ps, a_ps, r_ps = make_transformer(
                    state_dim=state_dim,
                    act_dim=act_dim,
                    n_blocks=n_blocks,
                    h_dim=h_dim,
                    context_len=context_len,
                    n_heads=n_heads,
                    drop_p=drop_p)(timesteps, states, actions, returns_to_go)
                return s_ps, a_ps, r_ps

        policy_module = PolicyModule()
        policy = FeedForwardModel(
            init=lambda key: policy_module.init(
                key, dummy_timesteps, dummy_states, dummy_actions, dummy_rtg),
            apply=policy_module.apply)
        return policy
    return policy_model_fn()
