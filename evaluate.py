import yaml
import numpy as np

from stable_baselines3 import PPO
from src.environment import CarRacingEnv

def load_config():
with open("config.yaml") as f:
return yaml.safe_load(f)

def main():

```
config = load_config()

env = CarRacingEnv(config)

model = PPO.load(config["paths"]["model_path"])

rewards = []

for _ in range(config["training"]["eval_episodes"]):

    obs, _ = env.reset()

    done = False
    episode_reward = 0

    while not done:

        action, _ = model.predict(obs)

        obs, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        episode_reward += reward

    rewards.append(episode_reward)

mean_reward = np.mean(rewards)
std_reward = np.std(rewards)

print(f"Mean Reward: {mean_reward:.2f}")
print(f"Std Reward: {std_reward:.2f}")
```

if **name** == "**main**":
main()
