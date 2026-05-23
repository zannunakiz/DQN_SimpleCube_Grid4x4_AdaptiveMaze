# ADAPTIVE DQN GRID MAZE

🎯 **DQN GRID MAZE** is an educational Python project that demonstrates a **Deep Q-Network (DQN)** agent inside a **4x4 GridWorld** environment. The repository also includes a lightweight visual editor to help design and inspect grid layouts manually.

## ✨ Overview

This project revolves around one main entry point:
- **`main.py`** for running DQN training and visualizing agent behavior.


The codebase is intentionally compact and readable, making it a good fit for:

- reinforcement learning demonstrations,
- thesis prototypes and early-stage experimentation,
- small controlled GridWorld studies with visual feedback.

## 🧠 Key Features

- Fixed start and goal positions.
- Random holes generated every episode, with path validation to guarantee at least one valid route to the goal.
- A compact 4-neuron local state representation: `up`, `down`, `left`, `right`.
- A simple fully connected DQN architecture.
- Optional real-time visualization through Pygame.

## 🗂️ Codebase Structure

- **`main.py`**  
  Contains the GridWorld environment, replay buffer, DQN network, agent logic, training loop, and CLI argument parsing.

- **`requirements.txt`**  
  Lists the core dependencies required to run the project.

## ⚙️ Requirements

- Python 3.10 or newer is recommended
- `pip` for dependency installation
- A desktop environment capable of opening Pygame windows

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd <filepath>

# Install project dependencies
pip install -r requirements.txt
```

## ▶️ Running the DQN Simulation

Use `main.py` to train the agent and optionally display the GridWorld window.

```bash
# Run training with default settings and GUI enabled
python main.py

# Train for 300 episodes
python main.py --episodes 300

# Limit each episode to 25 steps
python main.py --episodes 300 --max-steps 25

# Run training without opening the GUI
python main.py --no-render

# Show all available CLI options
python main.py --help
```

### 🔬 State Representation and Rewards

In `main.py`, the agent state is represented as:

```text
[up, down, left, right]
```

Sensor value meanings:

- `1` = white / clear cell
- `2` = yellow / previously visited cell
- `3` = red / hole
- `4` = wall / out of bounds
- `5` = green / goal

Action mapping:

- `0` = left
- `1` = right
- `2` = up
- `3` = down

Default rewards:

- `clear` = `+1`
- `yellow` = `-3`
- `wall` = `-10`
- `hole` = `-5`
- `goal` = `+20`

## 🔄 Training Flow

1. The environment resets and generates a valid random hole layout.
2. The agent selects actions using an epsilon-greedy policy.
3. Transitions are stored inside the replay buffer.
4. The online network learns from sampled experience batches.
5. The target network is updated periodically.
6. Episode logs are printed to the terminal during training.

## 📝 Notes

- This project is designed primarily for education and small-scale experimentation.
- If the Pygame window is closed during training, the program exits safely.
- Because the grid is intentionally small, the main focus is readability and experiment clarity rather than environment complexity.

## 👏 Credits

**Richky Abendego**  
Creator and Main Developer
