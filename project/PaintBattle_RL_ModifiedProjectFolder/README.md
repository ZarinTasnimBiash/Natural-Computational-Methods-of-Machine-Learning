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
```plaintext
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
```

---

## 🧠 How It Works

To enable agent-based control and training in the PaintBattle game, several key modifications were made to the original codebase. These adaptations were essential for allowing an external program to perceive game state and interact with the environment through actions. The major components of the system design are described in the [📄 Project Report](https://github.com/ZarinTasnimBiash/Natural-Computational-Methods-of-Machine-Learning/blob/main/project/Project_Report.pdf)

Next, we wanted to enable fast and efficient training of a DQL agent without the overhead of a visual game environment, and to refine the reward logic using improved density metrics, stride scaling, and memory-efficient state updates. Ergo, to accelerate the training process, a separate headless simulator was implemented. This simulator replicates the core mechanics of the original game—including player movement and painting logic while omitting colli- sions, pickups and user interface components. It maintains compatibility with the original game’s state and action interfaces, enabling faster training by eliminating visual overhead. For simplicity, the simulation environment includes only a single player. This setup allows the agent to focus on maximising canvas coverage without interference from other play- ers. While this does not capture the full dynamics of the multi-player game, it provides a predictable environment that helps shape an effective base behaviour. Its effectiveness in real multi-agent scenarios remains to be evaluated through further testing. Again, the major components of the system architecture are described in the [📄 Project Report](https://github.com/ZarinTasnimBiash/Natural-Computational-Methods-of-Machine-Learning/blob/main/project/Project_Report.pdf)
---

## 🧪 Running the Project

🔁 Headless Mode (Recommended for Training)

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

[Full academic report:](https://github.com/ZarinTasnimBiash/Natural-Computational-Methods-of-Machine-Learning/blob/main/project/Project_Report.pdf)

Includes:

Architecture diagrams
Reward analysis
Training results and plots



