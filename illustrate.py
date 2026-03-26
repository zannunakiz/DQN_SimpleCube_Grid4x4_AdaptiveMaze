"""
Interactive 4x4 grid illustrator for educational GridWorld experiments.

Features:
- Clickable 4x4 grid.
- Manual painting with semantic colors.
- Color picker panel on the right side.
- Keyboard shortcuts for faster editing and export to console.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame


GRID_SIZE = 4
CELL_SIZE = 110
GRID_LEFT = 20
GRID_TOP = 20
GRID_GAP = 4

PANEL_WIDTH = 260
WINDOW_WIDTH = GRID_LEFT * 2 + GRID_SIZE * CELL_SIZE + PANEL_WIDTH + 200
WINDOW_HEIGHT = GRID_TOP * 2 + GRID_SIZE * CELL_SIZE + 220

FPS = 60


COLORS = {
    "WHITE": (250, 250, 250),
    "YELLOW": (245, 211, 72),
    "BLUE": (67, 107, 229),
    "RED": (223, 73, 61),
    "GREEN": (83, 214, 105),
}

BG_COLOR = (236, 241, 247)
LINE_COLOR = (60, 60, 60)
TEXT_COLOR = (25, 25, 25)
PANEL_BG = (224, 231, 239)
SELECT_BORDER = (22, 22, 22)

# Labels rendered inside semantic cell colors.
LABELS = {
    "GREEN": "Goal",
    "BLUE": "Agent",
    "RED": "Hole",
}


def make_grid(default_color: str = "WHITE") -> list[list[str]]:
    return [[default_color for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]


def cell_rect(col: int, row: int) -> pygame.Rect:
    x = GRID_LEFT + col * CELL_SIZE
    y = GRID_TOP + row * CELL_SIZE
    return pygame.Rect(x, y, CELL_SIZE - GRID_GAP, CELL_SIZE - GRID_GAP)


def color_button_rect(index: int) -> pygame.Rect:
    panel_left = GRID_LEFT * 2 + GRID_SIZE * CELL_SIZE
    x = panel_left + 24
    y = GRID_TOP + 60 + index * 78
    return pygame.Rect(x, y, PANEL_WIDTH - 48, 58)


def draw_ui(
    screen: pygame.Surface,
    font_title: pygame.font.Font,
    font_body: pygame.font.Font,
    grid: list[list[str]],
    selected_color: str,
) -> None:
    screen.fill(BG_COLOR)

    # Draw the interactive grid.
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            rect = cell_rect(col, row)
            color_name = grid[row][col]
            pygame.draw.rect(screen, COLORS[color_name], rect)
            pygame.draw.rect(screen, LINE_COLOR, rect, 2)

            # Render centered labels for cells with a fixed semantic meaning.
            if color_name in LABELS:
                label_text = LABELS[color_name]
                label_surf = font_body.render(label_text, True, (0, 0, 0))
                label_rect = label_surf.get_rect(center=rect.center)
                screen.blit(label_surf, label_rect)

    # Draw the control panel.
    panel_left = GRID_LEFT * 2 + GRID_SIZE * CELL_SIZE
    panel_rect = pygame.Rect(panel_left, 0, PANEL_WIDTH, WINDOW_HEIGHT)
    pygame.draw.rect(screen, PANEL_BG, panel_rect)

    title = font_title.render("Color Picker", True, TEXT_COLOR)
    screen.blit(title, (panel_left + 24, GRID_TOP + 14))

    color_order = ["WHITE", "YELLOW", "BLUE", "RED", "GREEN"]
    for i, color_name in enumerate(color_order):
        rect = color_button_rect(i)
        pygame.draw.rect(screen, COLORS[color_name], rect, border_radius=8)
        pygame.draw.rect(screen, LINE_COLOR, rect, 2, border_radius=8)

        if color_name == selected_color:
            highlight = rect.inflate(8, 8)
            pygame.draw.rect(screen, SELECT_BORDER, highlight, 3, border_radius=10)

        text = font_body.render(f"{i + 1}. {color_name}", True, TEXT_COLOR)
        screen.blit(text, (rect.x + 12, rect.y + 16))

    lines = [
        "Click a color button,",
        "then click a cell on the grid.",
        "",
        "Shortcuts:",
        "1/2/3/4/5 select color",
        "C clear grid",
        "S print color matrix",
        "ESC exit",
    ]

    info_y = GRID_TOP + 390
    for line in lines:
        surf = font_body.render(line, True, TEXT_COLOR)
        screen.blit(surf, (panel_left + 24, info_y))
        info_y += 24


def grid_pos_from_mouse(pos: tuple[int, int]) -> tuple[int, int] | None:
    mx, my = pos
    if mx < GRID_LEFT or my < GRID_TOP:
        return None

    rel_x = mx - GRID_LEFT
    rel_y = my - GRID_TOP

    col = rel_x // CELL_SIZE
    row = rel_y // CELL_SIZE

    if 0 <= col < GRID_SIZE and 0 <= row < GRID_SIZE:
        return int(col), int(row)
    return None


def get_clicked_color(pos: tuple[int, int]) -> str | None:
    color_order = ["WHITE", "YELLOW", "BLUE", "RED", "GREEN"]
    for i, color_name in enumerate(color_order):
        if color_button_rect(i).collidepoint(pos):
            return color_name
    return None


def print_code_matrix(grid: list[list[str]]) -> None:
    print("\nCurrent 4x4 color matrix:")
    for row in grid:
        print(row)
    print("")


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Grid 4x4 Color Illustrator")

    font_title = pygame.font.SysFont("consolas", 28, bold=True)
    font_body = pygame.font.SysFont("consolas", 22)
    clock = pygame.time.Clock()

    grid = make_grid("WHITE")
    selected_color = "YELLOW"

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_1:
                    selected_color = "WHITE"
                elif event.key == pygame.K_2:
                    selected_color = "YELLOW"
                elif event.key == pygame.K_3:
                    selected_color = "BLUE"
                elif event.key == pygame.K_4:
                    selected_color = "RED"
                elif event.key == pygame.K_5:
                    selected_color = "GREEN"
                elif event.key == pygame.K_c:
                    grid = make_grid("WHITE")
                elif event.key == pygame.K_s:
                    print_code_matrix(grid)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                clicked_color = get_clicked_color(event.pos)
                if clicked_color is not None:
                    selected_color = clicked_color
                else:
                    grid_pos = grid_pos_from_mouse(event.pos)
                    if grid_pos is not None:
                        col, row = grid_pos
                        grid[row][col] = selected_color

        draw_ui(screen, font_title, font_body, grid, selected_color)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
