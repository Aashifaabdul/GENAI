# 🌙 Lunar Lander with Deep Q-Learning

Welcome to my dreamy descent into reinforcement learning! In this project, I taught an AI agent how to land safely on the moon using **Deep Q-Learning (DQN)**. With every crash and bounce, the agent learned to land with more grace—eventually pulling off the perfect soft landing like a seasoned space cadet 🚀💫


## 🚧 How the Lunar Lander Works

The environment is `LunarLander-v3` from OpenAI Gym.  
Imagine a tiny lander drifting in the silence of space, trying to land softly on a moon pad between two flags 🚩🚩

Here's how it plays out:

- The lander starts in the air with random position and velocity.
- The environment provides an **8-dimensional state vector** at each time step:
  - X & Y position
  - X & Y linear velocity
  - Lander's angle (orientation)
  - Angular velocity (rate of rotation)
  - Two boolean flags indicating if each leg is in contact with the ground
- It can take **4 actions**:
  - Do nothing
  - Fire left engine
  - Fire main engine (downward thrust)
  - Fire right engine

### The mission:
Safely land the module **between the flags** with minimal tilt and velocity.

Rewards:
- +100 for a successful landing
- -100 if it crashes
- Small penalty every timestep to encourage faster landings
- Bonus for leg contact

### 🧠 Training the Agent

1. An agent observes the state and picks an action using a neural network (Q-network).
2. Each interaction (state, action, reward, next_state, done) is stored in memory.
3. The agent learns by sampling past experiences (experience replay).
4. Two Q-networks are used—**local** and **target**—to stabilize training.
5. Training ends when the average reward over the last 100 episodes exceeds 200—meaning the agent can **consistently land like a boss** 🛬

https://github.com/user-attachments/assets/6a6bb44c-8207-4112-a438-ad316da5eb24


   
## 📁 Technologies used 
- Python 🐍  
- PyTorch 🧠  
- OpenAI Gym 🚀  
- NumPy 📊  
- Matplotlib 📈 


