import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from src.environment import CarRacingEnv

def load_config():
    # Fallback to config.yaml if no env var is set
    import os
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

class TrainingLoggerCallback(BaseCallback):
    def __init__(self, config, verbose=0):
        super().__init__(verbose)
        self.config = config
        self.timesteps = []
        self.mean_rewards = []
        self.rewards_buffer = []

    def _on_step(self) -> bool:
        # CRITICAL: Collect the reward from the current step
        # self.locals["rewards"] is a numpy array of rewards for all envs
        if "rewards" in self.locals:
            self.rewards_buffer.extend(self.locals["rewards"])

        # Log at specified intervals
        log_interval = self.config["training"]["log_interval"]
        if self.num_timesteps % log_interval == 0:
            mean_r = np.mean(self.rewards_buffer) if self.rewards_buffer else 0
            self.timesteps.append(self.num_timesteps)
            self.mean_rewards.append(float(mean_r))
            
            if self.verbose > 0:
                print(f"Step: {self.num_timesteps} - Mean Reward: {mean_r:.2f}")
            
            self.rewards_buffer = [] # Reset buffer for next interval
        return True

    def _on_training_end(self):
        # 1. Save JSON Log (Rubric Requirement)
        log_path = self.config["paths"]["training_log"]
        with open(log_path, "w") as f:
            json.dump({
                "timesteps": self.timesteps,
                "mean_rewards": self.mean_rewards
            }, f, indent=4)

        # 2. Generate Reward Plot (Rubric Requirement)
        plt.figure(figsize=(10, 5))
        plt.plot(self.timesteps, self.mean_rewards, label="Mean Reward")
        plt.xlabel("Timesteps")
        plt.ylabel("Reward")
        plt.title("Learning Progress")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.config["paths"]["reward_plot"])
        plt.close()

def main():
    config = load_config()
    env = CarRacingEnv(config)

    # Instantiate PPO with params from config
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        **config["ppo_params"]
    )

    callback = TrainingLoggerCallback(config, verbose=1)

    # Train
    model.learn(
        total_timesteps=config["training"]["total_timesteps"],
        callback=callback
    )

    # Save the final model (Rubric Requirement)
    model.save(config["paths"]["model_path"])

if __name__ == "__main__":
    main()