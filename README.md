# ADAPTIVE CUBE MAZE

🎯 **SKRPSI_CUBE** is an educational Python project that demonstrates a **Deep Q-Network (DQN)** agent inside a **4x4 GridWorld** environment. The repository also includes a lightweight visual editor to help design and inspect grid layouts manually.

## ✨ Overview

This project revolves around two main entry points:

- **`main.py`** for running DQN training and visualizing agent behavior.
- **`illustrate.py`** for opening an interactive 4x4 grid editor built with Pygame.

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
- A manual grid illustrator for creating and previewing board layouts.

## 🗂️ Codebase Structure

- **`main.py`**  
  Contains the GridWorld environment, replay buffer, DQN network, agent logic, training loop, and CLI argument parsing.

- **`illustrate.py`**  
  Provides an interactive 4x4 coloring tool for manually sketching grid scenarios with mouse input and keyboard shortcuts.

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
cd SKRPSI_CUBE

# Optional: create a virtual environment
python -m venv .venv

# Activate it in Windows PowerShell
.venv\Scripts\Activate.ps1

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

# Render every 3 steps to reduce GUI update frequency
python main.py --render-every 3 --fps 10

# Run training without opening the GUI
python main.py --no-render

# Show all available CLI options
python main.py --help
```

### CLI Options in `main.py`

- `--episodes` : number of training episodes to run.
- `--max-steps` : maximum number of steps allowed per episode.
- `--render-every` : render interval in steps when GUI mode is enabled.
- `--fps` : maximum frame rate for the Pygame renderer.
- `--seed` : random seed for reproducible experiments.
- `--no-render` : disable the GUI and run training in terminal-only mode.

## 🎨 Running the Grid Illustrator

Use `illustrate.py` to create a manual visual representation of a 4x4 grid.

```bash
# Open the interactive 4x4 grid editor
python illustrate.py
```

### Controls in `illustrate.py`

- Click a color button in the right panel, then click a cell on the grid.
- Press `1` for white.
- Press `2` for yellow.
- Press `3` for blue.
- Press `4` for red.
- Press `5` for green.
- Press `C` to clear the grid.
- Press `S` to print the current color matrix to the terminal.
- Press `Esc` to close the application.

## 🔬 State Representation and Rewards

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
