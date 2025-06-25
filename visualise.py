import gymnasium
import numpy as np
import time

env = gymnasium.make(
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
)

observation, info = env.reset(seed=42)  # Fixed seed for reproducibility

# Run for a fixed number of steps
for step in range(1000):
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    env.render()

    time.sleep(0.05)

    if terminated or truncated:
        observation, info = env.reset()
env.close()