import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Callable, NamedTuple

class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(64)(x))
        x = nn.relu(nn.Dense(64)(x))
        mean = nn.Dense(self.action_dim)(x)
        log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        std = jnp.exp(log_std)
        value = nn.Dense(1)(x)
        return mean, std, value.squeeze(-1)


class PPOState(NamedTuple):
    params: dict
    opt_state: optax.OptState

def make_ppo_fn(model: nn.Module, action_dim: int, lr: float = 3e-4):
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(lr)
    )

    def init(rng, obs_dim):
        dummy_obs = jnp.zeros((1, obs_dim))
        params = model.init(rng, dummy_obs)
        opt_state = optimizer.init(params)
        return PPOState(params, opt_state)

    def get_action(params, obs, rng):
        mean, std, _ = model.apply(params, obs)
        action = mean + std * jax.random.normal(rng, mean.shape)
        return action, mean, std


    def loss_fn(params, obs, actions, old_log_probs, advantages, returns):
        mean, std, values = model.apply(params, obs)
        log_std = jnp.log(std + 1e-8)

        # Log probability of action under Gaussian
        var = std ** 2
        log_probs = -0.5 * (((actions - mean) ** 2) / (var + 1e-8) + 2 * log_std + jnp.log(2 * jnp.pi))
        log_probs = log_probs.sum(axis=-1)

        ratio = jnp.exp(log_probs - old_log_probs)
        clipped_ratio = jnp.clip(ratio, 1 - 0.2, 1 + 0.2)
        actor_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))

        critic_loss = jnp.mean((returns - values) ** 2)
        entropy = jnp.mean(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))

        total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        return total_loss


    def update_fn(state, batch):
        grads = jax.grad(loss_fn)(state.params, *batch)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        return PPOState(new_params, opt_state)

    return init, get_action, update_fn
