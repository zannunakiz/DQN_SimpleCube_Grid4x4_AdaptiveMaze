"""
Implementasi edukatif Deep Q-Network (DQN) untuk GridWorld 4x4.

Karakteristik utama:
- Titik start dan goal selalu tetap.
- Hole diacak setiap episode sesuai `hole_count` dan selalu divalidasi.
- State agent hanya terdiri dari 4 neuron lokal: [atas, bawah, kiri, kanan].
- Aksi agent bersifat diskret: kiri, kanan, atas, dan bawah.
- Reward dibedakan untuk sel baru, sel terkunjungi, tembok, hole, dan goal.
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
from collections import deque
from typing import Deque, Dict, List, Optional, Set, Tuple

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim


class GridWorldEnv:
    """
    Grid 4x4 dengan start/goal tetap.

    Coordinate system:
    - (0, 0) = kiri atas
    - x bertambah ke kanan
    - y bertambah ke bawah
    """

    # Urutan aksi yang dipakai agent dan log training.
    # 0 = kiri, 1 = kanan, 2 = atas, 3 = bawah
    ACTIONS = {
        0: (-1, 0),  # kiri
        1: (1, 0),   # kanan
        2: (0, -1),  # atas
        3: (0, 1),   # bawah
    }

    # Urutan sensor lokal untuk 4 input neuron:
    # [atas, bawah, kiri, kanan]
    SENSOR_DIRS = [
        (0, -1),  # atas
        (0, 1),   # bawah
        (-1, 0),  # kiri
        (1, 0),   # kanan
    ]

    def __init__(
        self,
        grid_size: int = 4,
        start_pos: Tuple[int, int] = (0, 3),
        goal_pos: Tuple[int, int] = (3, 0),
        hole_count: int = 2,
        max_steps: int = 20,
        reward_clear: float = 1.0,
        reward_yellow: float = -3.0,
        reward_wall: float = -10.0,
        reward_hole: float = -5.0,
        reward_goal: float = 20.0,
        cell_size: int = 95,
        margin: int = 10,
        info_height: int = 140,
    ) -> None:
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.hole_count = hole_count
        self.max_steps = max_steps

        # Konfigurasi reward per jenis hasil langkah.
        self.reward_clear = reward_clear
        self.reward_yellow = reward_yellow
        self.reward_wall = reward_wall
        self.reward_hole = reward_hole
        self.reward_goal = reward_goal

        # Konfigurasi rendering Pygame.
        self.cell_size = cell_size
        self.margin = margin
        self.info_height = info_height

        self.agent_pos: Tuple[int, int] = self.start_pos
        self.holes: List[Tuple[int, int]] = []
        self.visited_cells: Set[Tuple[int, int]] = {self.start_pos}
        self.steps = 0

    @property
    def action_size(self) -> int:
        return 4

    @property
    def state_size(self) -> int:
        # Hanya 4 neuron input lokal: atas, bawah, kiri, kanan.
        return 4

    @property
    def screen_size(self) -> Tuple[int, int]:
        width = self.grid_size * self.cell_size + 2 * self.margin
        height = self.grid_size * self.cell_size + 2 * self.margin + self.info_height
        return width, height

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _has_possible_route(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        blocked: Set[Tuple[int, int]],
    ) -> bool:
        # Gunakan BFS sederhana untuk memastikan masih ada rute start -> goal.
        queue: Deque[Tuple[int, int]] = deque([start])
        visited = {start}

        while queue:
            x, y = queue.popleft()
            if (x, y) == goal:
                return True

            for dx, dy in self.ACTIONS.values():
                nx, ny = x + dx, y + dy
                npair = (nx, ny)
                if not self._in_bounds(nx, ny):
                    continue
                if npair in blocked:
                    continue
                if npair in visited:
                    continue
                visited.add(npair)
                queue.append(npair)

        return False

    def _generate_valid_holes(self) -> List[Tuple[int, int]]:
        """
        Buat layout hole acak yang tetap menyisakan jalur valid ke goal.
        """
        candidates = [
            (x, y)
            for y in range(self.grid_size)
            for x in range(self.grid_size)
            if (x, y) != self.start_pos and (x, y) != self.goal_pos
        ]

        # Ulangi sampling sampai menemukan kombinasi yang valid.
        # Untuk grid kecil seperti ini prosesnya biasanya sangat cepat.
        for _ in range(2000):
            holes = random.sample(candidates, self.hole_count)
            blocked = set(holes)
            if self._has_possible_route(self.start_pos, self.goal_pos, blocked):
                return holes

        # Fallback deterministik bila sampling acak belum menemukan hasil.
        for comb in itertools.combinations(candidates, self.hole_count):
            blocked = set(comb)
            if self._has_possible_route(self.start_pos, self.goal_pos, blocked):
                return list(comb)

        # Hampir tidak mungkin terjadi pada grid 4x4, tetapi tetap dijaga.
        raise RuntimeError("Tidak ditemukan layout hole valid dengan possible route.")

    def reset(self) -> np.ndarray:
        self.agent_pos = self.start_pos
        self.holes = self._generate_valid_holes()
        self.visited_cells = {self.start_pos}
        self.steps = 0
        return self._get_state_vector()

    def _get_state_vector(self) -> np.ndarray:
        """
        State 4-neuron:
        [atas, bawah, kiri, kanan]
        Value:
        - 1: putih (clear)
        - 2: kuning (visited)
        - 3: merah (hole)
        - 4: wall (out of bounds)
        - 5: hijau (goal)
        """
        ax, ay = self.agent_pos
        values: List[float] = []
        hole_set = set(self.holes)

        for dx, dy in self.SENSOR_DIRS:
            nx, ny = ax + dx, ay + dy
            if not self._in_bounds(nx, ny):
                values.append(4.0)
            elif (nx, ny) == self.goal_pos:
                values.append(5.0)
            elif (nx, ny) in hole_set:
                values.append(3.0)
            elif (nx, ny) in self.visited_cells:
                values.append(2.0)
            else:
                values.append(1.0)

        return np.array(values, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, str]]:
        self.steps += 1

        dx, dy = self.ACTIONS[action]
        x, y = self.agent_pos
        nx, ny = x + dx, y + dy

        # 1. Menabrak tembok -> penalti + terminal.
        if not self._in_bounds(nx, ny):
            return self._get_state_vector(), self.reward_wall, True, {"result": "wall"}

        # Pindahkan agent bila langkah masih valid.
        self.agent_pos = (nx, ny)

        # 2. Masuk hole -> penalti + terminal.
        if self.agent_pos in self.holes:
            return self._get_state_vector(), self.reward_hole, True, {"result": "hole"}

        # 3. Mencapai goal -> reward positif + terminal.
        if self.agent_pos == self.goal_pos:
            return self._get_state_vector(), self.reward_goal, True, {"result": "goal"}

        # 4. Masuk sel kosong atau sel yang pernah dikunjungi -> lanjut episode.
        done = False
        if self.agent_pos in self.visited_cells:
            result = "yellow"
            reward = self.reward_yellow
        else:
            result = "clear"
            reward = self.reward_clear

        self.visited_cells.add(self.agent_pos)

        # Hentikan episode jika batas langkah tercapai.
        if self.steps >= self.max_steps:
            done = True
            result = "max_steps"

        return self._get_state_vector(), reward, done, {"result": result}

    def render(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
        episode: int,
        total_reward: float,
        epsilon: float,
        last_result: str,
    ) -> None:
        screen.fill((236, 242, 248))

        hole_set = set(self.holes)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                rect = pygame.Rect(
                    self.margin + x * self.cell_size,
                    self.margin + y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )

                # Warna default untuk sel kosong.
                color = (250, 250, 250)

                if (x, y) == self.goal_pos:
                    color = (50, 182, 79)  # goal hijau
                elif (x, y) in hole_set:
                    color = (220, 70, 60)  # hole merah
                elif (x, y) in self.visited_cells:
                    color = (242, 210, 72)  # jalur yang pernah dilalui = kuning

                if (x, y) == self.agent_pos:
                    color = (65, 106, 230)  # agent biru

                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (72, 72, 72), rect, 2)

        info_top = self.margin * 2 + self.grid_size * self.cell_size
        lines = [
            "State input: [atas, bawah, kiri, kanan] -> 1 putih, 2 kuning, 3 merah, 4 wall, 5 hijau",
            f"Episode: {episode}",
            f"Total reward: {total_reward:.1f}",
            f"Epsilon: {epsilon:.3f}",
            f"Last event: {last_result}",
        ]
        for i, text in enumerate(lines):
            surf = font.render(text, True, (25, 25, 25))
            screen.blit(surf, (self.margin, info_top + i * 26))


class ReplayBuffer:
    def __init__(self, capacity: int = 5000) -> None:
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """
    Fully connected network tanpa CNN.

    Input  : 4 neuron state lokal.
    Output : 4 Q-values untuk kiri, kanan, atas, dan bawah.
    """

    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        gamma: float = 0.9,
        lr: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        target_update_every: int = 100,
        replay_capacity: int = 5000,
        device: str = "cpu",
    ) -> None:
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.device = torch.device(device)

        self.memory = ReplayBuffer(replay_capacity)
        self.online_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.learn_steps = 0

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.add(state, action, reward, next_state, done)

    def learn(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Estimasi Q(s, a) saat ini dari online network.
        current_q = self.online_net(states_t).gather(1, actions_t).squeeze(1)

        # Hitung Bellman target menggunakan target network.
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1).values
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def process_pygame_events() -> bool:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
    return True


def maybe_render(
    env: GridWorldEnv,
    screen: Optional[pygame.Surface],
    font: Optional[pygame.font.Font],
    clock: Optional[pygame.time.Clock],
    should_render: bool,
    fps: int,
    episode: int,
    total_reward: float,
    epsilon: float,
    last_result: str,
) -> None:
    if not should_render or screen is None or font is None or clock is None:
        return
    env.render(screen, font, episode, total_reward, epsilon, last_result)
    pygame.display.flip()
    clock.tick(fps)


def train_dqn(
    env: GridWorldEnv,
    episodes: int,
    render: bool,
    render_every: int,
    fps: int,
) -> List[float]:
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        gamma=0.9,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_min=0.001,
        epsilon_decay=0.995,
        batch_size=32,
        target_update_every=100,
        replay_capacity=5000,
        device="cpu",
    )

    rewards_history: List[float] = []

    screen = None
    font = None
    clock = None
    if render:
        pygame.init()
        screen = pygame.display.set_mode(env.screen_size)
        pygame.display.set_caption("DQN GridWorld (Fixed Start/Goal + Random Holes)")
        font = pygame.font.SysFont("consolas", 22)
        clock = pygame.time.Clock()

    running = True
    global_step = 0
    render_every = max(1, render_every)
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0
        losses: List[float] = []
        last_result = "reset"

        while not done:
            if render and not process_pygame_events():
                running = False
                break

            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward
            last_result = info["result"]
            global_step += 1

            # Render sinkron:
            # - setiap N step (`render_every`)
            # - selalu render saat terminal agar hasil episode terlihat di GUI
            render_now = render and ((global_step % render_every == 0) or done)
            maybe_render(
                env=env,
                screen=screen,
                font=font,
                clock=clock,
                should_render=render_now,
                fps=fps,
                episode=episode,
                total_reward=total_reward,
                epsilon=agent.epsilon,
                last_result=last_result,
            )

        if not running:
            break

        agent.decay_epsilon()
        rewards_history.append(total_reward)

        avg = float(np.mean(rewards_history))
        avg_loss = float(np.mean(losses)) if losses else float("nan")
        loss_text = f"{avg_loss:.4f}" if not np.isnan(avg_loss) else "n/a"
        holes_text = "".join(f"({hx},{hy})" for hx, hy in sorted(env.holes))
        status = "SUCCESS" if last_result == "goal" else "FAIL"
        print(
            f"Episode {episode:4d} | {status:7s} | End {last_result:9s} | "
            f"Hole {holes_text} | Step {env.steps:3d} | Reward {total_reward:6.1f} | Avg {avg:6.2f} | "
            f"Eps {agent.epsilon:.3f} | Loss {loss_text}"
        )

    if render:
        pygame.quit()

    return rewards_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DQN GridWorld edukasi dengan state lokal 4-neuron"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=400,
        help="Jumlah episode training yang dijalankan",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maksimum langkah per episode",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="Render setiap N step saat GUI aktif",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="Batas frame per second untuk renderer Pygame",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed agar eksperimen dapat direproduksi",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Jalankan training tanpa membuka jendela Pygame",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    env = GridWorldEnv(
        grid_size=4,
        start_pos=(0, 3),   # posisi start tetap
        goal_pos=(3, 0),    # posisi goal tetap
        hole_count=2,       # dua hole acak setiap episode
        max_steps=args.max_steps,
        reward_clear=1.0,
        reward_yellow=-3.0,
        reward_wall=-10.0,
        reward_hole=-5.0,
        reward_goal=20.0,
    )

    print("Konfigurasi skenario:")
    print("- Grid         : 4x4")
    print("- Start tetap  : (0, 3)")
    print("- Goal tetap   : (3, 0)")
    print("- Hole per eps : 2 (acak, tapi selalu ada possible route)")
    print(f"- Episodes     : {args.episodes}")
    print("- State input  : [atas, bawah, kiri, kanan] -> 1(putih), 2(kuning), 3(merah), 4(wall), 5(hijau)")
    print("- Aksi         : 0=kiri, 1=kanan, 2=atas, 3=bawah")
    print("- Reward       : clear=+1, yellow=-3, wall=-10, hole=-5, goal=+20")
    print("")

    rewards = train_dqn(
        env=env,
        episodes=args.episodes,
        render=not args.no_render,
        render_every=args.render_every,
        fps=args.fps,
    )

    if rewards:
        print(f"\nFinal Avg Reward: {np.mean(rewards):.2f}")


if __name__ == "__main__":
    main()
