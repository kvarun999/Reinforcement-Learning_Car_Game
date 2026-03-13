# 🏎️ Reinforcement Learning Car Game

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Stable Baselines3](https://img.shields.io/badge/RL-Stable%20Baselines3-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![Pygame](https://img.shields.io/badge/UI-Pygame-green.svg)](https://www.pygame.org/)
[![Gymnasium](https://img.shields.io/badge/Env-Gymnasium-yellow.svg)](https://gymnasium.farama.org/)

An autonomous driving agent trained using Reinforcement Learning (PPO) to navigate a 2D track using LIDAR-like sensor data. The environment is built with Pygame and wrapped in a Gymnasium interface.

---

## 🌟 Features

- **Sensor-Based Navigation**: The car uses ray-casting (LIDAR logic) to perceive its environment.
- **Custom Gymnasium Environment**: Implements a `CarRacingEnv` compatible with standard RL algorithms.
- **State-of-the-Art RL Training**: Uses the Proximal Policy Optimization (PPO) algorithm via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/).
- **MLOps-Ready Containerization**: Fully dockerized environment for seamless deployment and training.
- **Automated Logging**: Plots reward curves over time and logs training metrics in JSON format.
- **Video Recording**: Save demonstrations of the fully-trained agent performing on the track.

---

## 📂 Project Structure

```text
/
├── Dockerfile              # Docker configuration for MLOps
├── requirements.txt        # Python dependencies
├── config.yaml             # Hyperparameters and environment configuration
├── train.py                # Main script to train the RL agent
├── evaluate.py             # Script to evaluate real-time performance of a model
├── record_video.py         # Script to record the agent's gameplay to a video file
├── src/                    # Source code for the core environment and agent logic
│   ├── car.py              # Car physics and sensor mechanics (Ray-casting)
│   ├── environment.py      # Gymnasium-compatible environment wrapper
│   └── utils.py            # Helper functions for math and Pygame operations
├── tracks/                 # Track layouts and files
│   └── track_1.txt
├── models/                 # Saved Stable-Baselines3 models (e.g., .zip files)
└── results/                # Output graphs, videos, and JSON logs
```

---

## 🚀 Setup and Installation

### Local Installation 💻
1. **Clone the repository** (or download the source code):
2. **Install dependencies**:
   Make sure you have a working Python 3.8+ environment.
   ```bash
   pip install -r requirements.txt
   ```

### Docker Installation 🐳 (MLOps Ready)
To build and run the project using Docker:
```bash
docker build -t rl-car-game .
docker run -it rl-car-game
```

---

## 🎮 Usage 

### 1. Training the Model
To train the agent using the PPO algorithm:
```bash
python train.py
```
*The training script will output models and logs into the `models/` and `results/` directories based on the settings in `config.yaml`.*

### 2. Evaluating the Model
Watch your trained agent perform in the environment:
```bash
python evaluate.py
```

### 3. Recording a Demonstration
Save a video of the trained agent navigating the track:
```bash
python record_video.py
```

---

## ⚙️ Configuration
All hyperparameters, reward definitions, and paths are managed via the `config.yaml` file. Key parameters you can tune include:

- `learning_rate`: Step size for the PPO optimizer.
- `total_timesteps`: The total number of steps the agent gets to interact and learn.
- `num_rays`: The number of LIDAR-like rays (sensors) the car uses.
- `rewards`: Customize the collision penalties and survival rewards.

---

## 📄 License
This project is for educational purposes as part of an MLOps and Reinforcement Learning curriculum. Feel free to use, modify, and learn from it.
