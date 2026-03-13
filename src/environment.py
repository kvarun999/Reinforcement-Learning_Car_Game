import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.car import Car
from src.utils import load_track


class CarRacingEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, config: dict, render_mode=None):
        super().__init__()

        self.config = config
        self.render_mode = render_mode

        env_params = config["env_params"]

        # Load track
        self.walls = load_track(config["paths"]["track_file"])

        # Initialize car
        self.car = Car(config)

        # Observation = ray distances + velocity
        obs_size = env_params["num_rays"] + 1

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )

        # Actions:
        # 0 = no-op
        # 1 = accelerate
        # 2 = brake
        # 3 = turn left
        # 4 = turn right
        self.action_space = spaces.Discrete(5)

        self.max_steps = env_params["max_steps_per_episode"]
        self.step_penalty = float(env_params["step_penalty"])
        self.collision_penalty = float(env_params["collision_penalty"])

        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Pull from config to avoid hardcoding
        self.car.x = self.config["env_params"].get("start_x", 100.0)
        self.car.y = self.config["env_params"].get("start_y", 100.0)
        self.car.angle = self.config["env_params"].get("start_angle", 0.0)
        self.car.velocity = 0.0

        self.step_count = 0
        rays = self.car.cast_rays(self.walls)
        velocity = self.car.velocity / self.car.max_velocity
        obs = np.array(rays + [velocity], dtype=np.float32)

        return obs, {}

    def step(self, action):

        # Apply action
        if action == 1:
            self.car.accelerate()

        elif action == 2:
            self.car.brake()

        elif action == 3:
            self.car.turn(-1)

        elif action == 4:
            self.car.turn(1)

        # Update physics
        self.car.update()

        # Get observation
        rays = self.car.cast_rays(self.walls)

        velocity = self.car.velocity / self.car.max_velocity

        obs = np.array(rays + [velocity], dtype=np.float32)

        # Reward calculation
        reward = self.step_penalty

        # Collision detection (ray-based approximation)
        collision = min(rays) < 0.02

        terminated = False

        if collision:
            reward += self.collision_penalty
            terminated = True

        # Episode truncation
        self.step_count += 1

        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def render(self):
        # Rendering will be implemented later
        pass

    def close(self):
        pass