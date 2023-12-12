import pygame
import neat
import random
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.Font("arial.ttf", 25)

# Global Information class containing all variables
class global_information:
    BLOCK_SIZE = 20
    WIDTH = 20
    HEIGHT = 20
    SPEED = 60
    WIN_WIDTH = WIDTH * BLOCK_SIZE
    WIN_HEIGHT = HEIGHT * BLOCK_SIZE

    WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Snake Game")

    GEN = 0

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
    def __init__(self, generation, individual):
        self.generation = generation
        self.individual = individual
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

        if action == 0:
            self.direction = Direction.LEFT
        elif action == 1:
            self.direction = Direction.RIGHT
        elif action == 2:
            self.direction = Direction.UP
        elif action == 3:
            self.direction = Direction.DOWN

        # 2. move
        self._move(self.direction)
        self.snake.insert(0, self.head)

        # 3. check if gameover
        game_over = False
        if self._is_collide() or self.frame_iteration > 50 * len(self.snake):
            game_over = True
            return game_over, self.score

        self.score += 2

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(GLOBAL_INFO.SPEED)
        # 6. return game over and score
        return game_over, self.score

    def _update_ui(self):
        GLOBAL_INFO.WIN.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(
                GLOBAL_INFO.WIN,
                BLUE1,
                pygame.Rect(pt.x, pt.y, GLOBAL_INFO.BLOCK_SIZE, GLOBAL_INFO.BLOCK_SIZE),
            )
            pygame.draw.rect(
                GLOBAL_INFO.WIN,
                BLUE2,
                pygame.Rect(
                    pt.x + 4,
                    pt.y + 4,
                    GLOBAL_INFO.BLOCK_SIZE - 8,
                    GLOBAL_INFO.BLOCK_SIZE - 8,
                ),
            )

        pygame.draw.rect(
            GLOBAL_INFO.WIN,
            RED,
            pygame.Rect(
                self.food.x, self.food.y, GLOBAL_INFO.BLOCK_SIZE, GLOBAL_INFO.BLOCK_SIZE
            ),
        )

        text = font.render("Score : " + str(self.score), True, WHITE)
        GLOBAL_INFO.WIN.blit(text, [0, 0])
        text = font.render("Gen : " + str(self.generation), True, WHITE)
        GLOBAL_INFO.WIN.blit(text, [0, 25])
        text = font.render("Individual : " + str(self.individual), True, WHITE)
        GLOBAL_INFO.WIN.blit(text, [0, 50])
        pygame.display.flip()

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += GLOBAL_INFO.BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= GLOBAL_INFO.BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += GLOBAL_INFO.BLOCK_SIZE
        elif direction == Direction.UP:
            y -= GLOBAL_INFO.BLOCK_SIZE

        self.head = Point(x, y)

    def _is_collide(self):
        # Solid Walls
        if (
            self.head.x > GLOBAL_INFO.WIN_WIDTH - GLOBAL_INFO.BLOCK_SIZE
            or self.head.x < 0
            or self.head.y > GLOBAL_INFO.WIN_HEIGHT - GLOBAL_INFO.BLOCK_SIZE
            or self.head.y < 0
        ):
            return True

        if self.head in self.snake[1:]:
            return True

        return False

    def get_inputs(self):
        # 1. Head (x/X, y/Y)
        x_head = self.head.x / GLOBAL_INFO.WIDTH
        y_head = self.head.y / GLOBAL_INFO.HEIGHT

        # 2. Food (x/X, y/Y)
        x_food = self.food.x / GLOBAL_INFO.WIDTH
        y_food = self.food.y / GLOBAL_INFO.HEIGHT

        # 3. Walls (left, right, straight)

        clockwise_direction = [
            Direction.RIGHT,
            Direction.DOWN,
            Direction.LEFT,
            Direction.UP,
        ]

        curr_direction = clockwise_direction.index(self.direction)

        dist_wall_right = self.head.x / GLOBAL_INFO.WIDTH
        dist_wall_down = self.head.y / GLOBAL_INFO.HEIGHT
        dist_wall_left = 1 - (self.head.x / GLOBAL_INFO.WIDTH)
        dist_wall_up = 1 - (self.head.y / GLOBAL_INFO.HEIGHT)

        clockwise_wall_distances = [
            dist_wall_right,
            dist_wall_down,
            dist_wall_left,
            dist_wall_up,
        ]

        straight_wall = clockwise_wall_distances[curr_direction]
        right_wall = clockwise_wall_distances[(curr_direction + 1) % 4]
        left_wall = clockwise_wall_distances[(curr_direction - 1) % 4]

        # 4. Body on all sides
        dist_body_right = 0
        dist_body_left = 0
        dist_body_up = 0
        dist_body_down = 0

        for pt in self.snake[1:]:
            # for right side
            if self.head.y == pt.y and self.head.x < pt.x:
                dist_body_right = max(dist_body_right, self.head.x / pt.x)

            # for left side
            if self.head.y == pt.y and self.head.x > pt.x:
                dist_body_left = max(dist_body_left, pt.x / self.head.x)

            # for up side
            if self.head.x == pt.x and self.head.y > pt.y:
                dist_body_up = max(dist_body_up, pt.y / self.head.y)

            # for down side
            if self.head.x == pt.x and self.head.y < pt.y:
                dist_body_down = max(dist_body_down, self.head.y / pt.y)

        clockwise_body_distances = [
            dist_body_right,
            dist_body_down,
            dist_body_left,
            dist_body_up,
        ]

        straight_body = clockwise_body_distances[curr_direction]
        right_body = clockwise_body_distances[(curr_direction + 1) % 4]
        left_body = clockwise_body_distances[(curr_direction - 1) % 4]

        # 5. Length (len/area)
        length = len(self.snake) / (GLOBAL_INFO.WIDTH * GLOBAL_INFO.HEIGHT)

        return (
            x_head,
            y_head,
            x_food,
            y_food,
            left_wall,
            straight_wall,
            right_wall,
            left_body,
            straight_body,
            right_body,
            length,
        )
