import yaml
import os
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
from src.environment import CarRacingEnv

def load_config():
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

    # Use rgb_array for video recording (mandatory for headless/Docker)
    env = CarRacingEnv(config, render_mode="rgb_array")

    video_folder = "results"
    
    # Wrap the environment to record video (Requirement 6)
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda e: True, # Record every episode
        name_prefix="agent_demonstration",
        disable_logger=True
    )

    # Load the trained model (Requirement 6)
    model_path = config["paths"]["model_path"]
    model = PPO.load(model_path)

    obs, _ = env.reset()
    done = False

    while not done:
        # Use deterministic=True for the best visual demonstration
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    # Close and save the video
    env.close()
    
    print(f"Video recording complete. Check the {video_folder} directory.")

if __name__ == "__main__":
    main()