Paint Battle
=======

Forked from A1rPun to try some reinforcement learning on this game.

# [Play](https://A1rPun.github.io/PaintBattle)

**How to play**
- Player 1 is pink
- Use the left & right arrow keys to move

Inspired by the original game **Battle Painters**
![original.jpg](/img/original.jpg)

# 🎨 PaintBattleDQL: Deep Q-Learning Agent for a 2D Painting Game

**PaintBattleDQL** is a reinforcement learning project where an autonomous agent is trained using **Deep Q-Learning (DQL)** to play a 2D browser-based multiplayer painting game. The agent learns to **maximize canvas coverage** in a time-limited environment through intelligent navigation and strategic exploration.

> 🧠 Developed by Group 13 as part of the *Natural Computation Methods in Machine Learning (NCML)* course at Uppsala University, Spring 2025.

---

## 🚀 Project Highlights

- 🕹️ Real-time WebSocket communication between the browser game and Python agent
- 🧪 Fast headless training mode using a grid-based simulator
- 🧠 Deep Q-Network (DQN) with experience replay and target networks
- 🎯 Custom reward mechanism encouraging spatial exploration
- 📈 Saved model evaluation and training visualization

---

## 🧱 Project Structure
PaintBattleDQL/
├── websocket.py
├── dql.py
├── headless_battle_painters.py
├── headless_dql.py
├── index.html
├── training_coverage.csv
├── js/
│   ├── game.js
│   ├── gameState.js
│   ├── main.js
│   ├── object.js
│   ├── pickup.js
│   ├── rl-agent.js
│   └── vector.js
├── snd/
│   ├── game.mp3
│   └── mainmenu.mp3
├── img/
│   ├── brush.png
│   └── ... other .png files
├── css/
│   ├── main.css
│   └── main.less
├── battle_painter_model_315_64.pt      # Best model after training
├── annotated_group_13_paintbattle_dql-1.pdf
└── README.md


---

## 🧠 How It Works

### 1. WebSocket Integration

- Modified the PaintBattle game to allow control via WebSocket
- Agent sends actions (left, right, forward); game returns states
- Supports both human and agent play modes

### 2. RL Agent (DQL)

- Implemented in PyTorch (`dql.py`)
- 9-dimensional state vector (position, angle, canvas coverage, etc.)
- Uses ε-greedy strategy, replay buffer, and target Q-network

### 3. Headless Simulation (Fast Training)

- `headless_battle_painters.py` simulates painting logic using grids
- Faster training without rendering
- Reward = canvas coverage + density-based exploration

---

## 🧪 Running the Project

### 🔁 Headless Mode (Recommended for Training)

```bash
python headless_battle_painters.py  # Terminal 1

Keep it running

Then in a second terminal:
python headless_dql.py              # Terminal 2

🕸️ Browser Mode (Live Game)

1. Start WebSocket server:
python websocket.py

2. Open index.html in a browser

3. Run the agent:
python dql.py

🧪 Inference Mode

To run a trained agent:
python dql.py --mode inference --model_path models/best_model.pth

📚 Future Work

Multi-agent training (competitive/cooperative)
Temporal modeling with LSTMs
Double DQN / Dueling DQN
Power-ups, dynamic hazards
Hyperparameter tuning with Ray Tune

👥 Authors

Prakhar Dubey – RL agent, headless simulation, codebase structuring
Zarin Tasnim Biash – WebSocket integration, training setup, documentation

📄 Report

Full academic report: annotated-group_13_paintbattle_dql-1.pdf

Includes:

Architecture diagrams
Reward analysis

Training results and plots
