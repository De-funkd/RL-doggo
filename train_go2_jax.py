import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from ppo_agent import ActorCritic, make_ppo_fn

env = gym.make(
    'Ant-v5',
    xml_file='/home/ansh/projekts/RL-doggo/unitree_go2/scene.xml',  # Path to Go2 MJCF model
    forward_reward_weight=1.0,
    ctrl_cost_weight=0.05,
    contact_cost_weight=5e-4,
    healthy_reward=1.0,
    main_body=1,  # Assumes trunk is body 1; verify in model
    healthy_z_range=(0.25, 0.9),  # Adjusted for Go2 height
    include_cfrc_ext_in_observation=True,
    exclude_current_positions_from_observation=False,
    reset_noise_scale=0.1,
    frame_skip=25,  # dt = 0.05s (assuming model timestep=0.002s)
    max_episode_steps=1000,
    render_mode='human'  # Enable rendering
)  # Make sure this loads your MuJoCo go2.xml

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]

rng = jax.random.PRNGKey(0)
model = ActorCritic(action_dim=action_dim)
init_fn, act_fn, update_fn = make_ppo_fn(model, action_dim)

ppo_state = init_fn(rng, obs_dim)

NUM_UPDATES = 5000
ROLLOUT_LEN = 2048
BATCH_SIZE = 64

for update in range(NUM_UPDATES):
    obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
    obs, _ = env.reset()
    done = False

    for _ in range(ROLLOUT_LEN):
        obs_jax = jnp.array(obs[None])
        rng, subrng = jax.random.split(rng)
        action, mean, std = act_fn(ppo_state.params, obs_jax, subrng)
        logp = -0.5 * (((action - mean) / (std + 1e-8))**2 + 2 * jnp.log(std + 1e-8) + jnp.log(2 * jnp.pi))
        logp = jnp.sum(logp, axis=-1)


        next_obs, reward, terminated, truncated, _ = env.step(np.array(action))
        done = terminated or truncated
        val_estimate = model.apply(ppo_state.params, obs_jax)[1]

        obs_buf.append(obs)
        act_buf.append(action)
        logp_buf.append(logp)
        rew_buf.append(reward)
        val_buf.append(val_estimate)
        done_buf.append(done)

        obs = next_obs
        if done:
            obs = env.reset()

    # Convert to JAX arrays
    obs_buf = jnp.array(obs_buf)
    act_buf = jnp.array(act_buf)
    logp_buf = jnp.array(logp_buf)
    rew_buf = jnp.array(rew_buf)
    val_buf = jnp.array(val_buf)
    done_buf = jnp.array(done_buf)

    # Compute returns and advantages (GAE optional)
    returns = rew_buf  # Simplified; you can compute full GAE here
    advantages = returns - val_buf

    # Shuffle and batch
    idx = np.random.permutation(ROLLOUT_LEN)
    for start in range(0, ROLLOUT_LEN, BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_idx = idx[start:end]
        batch = (
            obs_buf[batch_idx],
            act_buf[batch_idx],
            logp_buf[batch_idx],
            advantages[batch_idx],
            returns[batch_idx]
        )
        ppo_state = update_fn(ppo_state, batch)

    print(f"Update {update}: avg reward = {jnp.mean(rew_buf):.2f}")
