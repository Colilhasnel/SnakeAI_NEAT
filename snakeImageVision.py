import pygame
import math
import random
import os
from enum import Enum
from collections import namedtuple
import numpy as np


pygame.init()
font = pygame.font.Font("arial.ttf", 25)


# Global Information class containing all variables
class global_information:
    BLOCK_SIZE = 16
    WIDTH = 32
    HEIGHT = 32
    SPEED = 150
    WIN_WIDTH = WIDTH * BLOCK_SIZE
    WIN_HEIGHT = HEIGHT * BLOCK_SIZE
    WIN_DIAGONAL = math.sqrt(WIN_WIDTH**2 + WIN_HEIGHT**2)
    ROOT2 = math.sqrt(2)
    DIAGONAL = math.sqrt(WIDTH**2 + HEIGHT**2)


# Instance of global_information to be used by all functions
GLOBAL_INFO = global_information()


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)


class Snake:
    def __init__(self):

        self.display = pygame.display.set_mode(
            (GLOBAL_INFO.WIN_WIDTH, GLOBAL_INFO.WIN_HEIGHT)
        )
        pygame.display.set_caption("Snake Game")

        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(GLOBAL_INFO.WIN_WIDTH / 2, GLOBAL_INFO.WIN_HEIGHT / 2)
        self.snake = [
            self.head,
            Point(self.head.x - GLOBAL_INFO.BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * GLOBAL_INFO.BLOCK_SIZE, self.head.y),
        ]

        self.score = 0
        self.food = None
        self.frame_iteration = 0

        self._place_food()

        self._update_ui()

    def _place_food(self):
        x = (
            random.randint(
                0,
                (GLOBAL_INFO.WIN_WIDTH - GLOBAL_INFO.BLOCK_SIZE)
                // GLOBAL_INFO.BLOCK_SIZE,
            )
            * GLOBAL_INFO.BLOCK_SIZE
        )
        y = (
            random.randint(
                0,
                (GLOBAL_INFO.WIN_HEIGHT - GLOBAL_INFO.BLOCK_SIZE)
                // GLOBAL_INFO.BLOCK_SIZE,
            )
            * GLOBAL_INFO.BLOCK_SIZE
        )
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1

        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. check if gameover
        reward = 0
        game_over = False
        if self._is_collide() or self.frame_iteration > 50 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            x_dist = (self.food.x - self.head.x) // 16
            y_dist = (self.food.y - self.head.y) // 16
            reward = math.sqrt(x_dist**2 + y_dist**2) / int(GLOBAL_INFO.DIAGONAL)
            reward = math.exp(-2 * reward)
            reward = reward / 100
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(GLOBAL_INFO.SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(
                self.display,
                BLUE1,
                pygame.Rect(pt.x, pt.y, GLOBAL_INFO.BLOCK_SIZE, GLOBAL_INFO.BLOCK_SIZE),
            )
            pygame.draw.rect(
                self.display,
                BLUE2,
                pygame.Rect(
                    pt.x + 4,
                    pt.y + 4,
                    GLOBAL_INFO.BLOCK_SIZE - 8,
                    GLOBAL_INFO.BLOCK_SIZE - 8,
                ),
            )

        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(
                self.food.x, self.food.y, GLOBAL_INFO.BLOCK_SIZE, GLOBAL_INFO.BLOCK_SIZE
            ),
        )

        text = font.render("Score : " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

        pygame.image.save(self.display, "ImageVision.jpg")

    def _move(self, action):

        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clockwise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clockwise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clockwise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += GLOBAL_INFO.BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= GLOBAL_INFO.BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += GLOBAL_INFO.BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= GLOBAL_INFO.BLOCK_SIZE

        self.head = Point(x, y)

    def _is_collide(self, pt=None):
        # Solid Walls
        if pt == None:
            pt = self.head
        if (
            pt.x > GLOBAL_INFO.WIN_WIDTH - GLOBAL_INFO.BLOCK_SIZE
            or pt.x < 0
            or pt.y > GLOBAL_INFO.WIN_HEIGHT - GLOBAL_INFO.BLOCK_SIZE
            or pt.y < 0
        ):
            return True

        if self.head in self.snake[1:]:
            return True

        return False
