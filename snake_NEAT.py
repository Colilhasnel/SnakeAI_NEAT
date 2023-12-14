import pygame
import math
import neat
import random
import os
from enum import Enum
from collections import namedtuple


pygame.init()
font = pygame.font.Font("arial.ttf", 25)


# Global Information class containing all variables
class global_information:
    BLOCK_SIZE = 20
    WIDTH = 10
    HEIGHT = 10
    SPEED = 120
    WIN_WIDTH = WIDTH * BLOCK_SIZE
    WIN_HEIGHT = HEIGHT * BLOCK_SIZE
    WIN_DIAGONAL = math.sqrt(WIN_WIDTH**2 + WIN_HEIGHT**2)
    ROOT2 = math.sqrt(2)
    DIAGONAL = math.sqrt(WIDTH**2 + HEIGHT**2)

    WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Snake Game")

    GEN = 0
    NO_SNAKES = 0
    AVG_LENGTH = 0


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
            self.direction = Direction.RIGHT
        elif action == 1:
            self.direction = Direction.DOWN
        elif action == 2:
            self.direction = Direction.LEFT
        elif action == 3:
            self.direction = Direction.UP

        # 2. move
        self._move(self.direction)
        self.snake.insert(0, self.head)

        # 3. check if gameover
        game_over = False
        if self._is_collide() or self.frame_iteration > 50 * len(self.snake):
            game_over = True
            return game_over, self.score, self.frame_iteration

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(GLOBAL_INFO.SPEED)
        # 6. return game over and score
        return game_over, self.score, self.frame_iteration

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

    def get_Vision(self):
        # RIGHT TO UP (V4 Directions)
        clockwise_direction = [
            Direction.RIGHT,
            Direction.DOWN,
            Direction.LEFT,
            Direction.UP,
        ]

        curr_direction_4 = clockwise_direction.index(self.direction)

        # V0 to V7 (V8 Directions)
        get_look_clockwise = lambda pt1, pt0=self.head: [
            ((pt0.x == pt1.x) and (pt1.y < pt0.y)),
            (
                ((pt0.y - pt1.y) == (pt1.x - pt0.x))
                and (pt1.x > pt0.x)
                and (pt1.y < pt0.y)
            ),
            ((pt1.y == pt0.y) and (pt1.x > pt0.x)),
            (
                ((pt1.x - pt0.x) == (pt1.y - pt0.y))
                and (pt1.x > pt0.x)
                and (pt1.y > pt0.y)
            ),
            ((pt1.x == pt0.x) and (pt1.y > pt0.y)),
            (
                ((pt0.y - pt1.y) == (pt1.x - pt0.x))
                and (pt1.x < pt0.x)
                and (pt1.y > pt0.y)
            ),
            ((pt1.y == pt0.y) and (pt1.x < pt0.x)),
            (
                ((pt1.x - pt0.x) == (pt1.y - pt0.y))
                and (pt1.x < pt0.x)
                and (pt1.y < pt0.y)
            ),
        ]

        # 1. Food Location (One-Hot Encoded) (V8 Look)
        food_inputs = get_look_clockwise(self.food)

        # # conversion of V4 to V8
        # curr_direction_8 = (2 * curr_direction_4 + 2) % 8

        # food_inputs = (
        #     clockwise_food[curr_direction_8:] + clockwise_food[:curr_direction_8]
        # )

        # 2. Body Location (One-Hot Encoded) (V8 Look)

        snake_body = [False] * 8

        for pt in self.snake[1:]:
            look = get_look_clockwise(pt)
            snake_body = [snake_body[i] | look[i] for i in range(0, 8)]

        # 3. Walls (Distances) (V8 Look)
        walls = [
            self.head.y / GLOBAL_INFO.WIN_HEIGHT,
            min(self.head.y, GLOBAL_INFO.WIN_WIDTH - self.head.x)
            * GLOBAL_INFO.ROOT2
            / GLOBAL_INFO.WIN_DIAGONAL,
            1 - (self.head.x / GLOBAL_INFO.WIN_WIDTH),
            min(
                GLOBAL_INFO.WIN_HEIGHT - self.head.y,
                GLOBAL_INFO.WIN_WIDTH - self.head.x,
            )
            * GLOBAL_INFO.ROOT2
            / GLOBAL_INFO.WIN_DIAGONAL,
            1 - (self.head.y / GLOBAL_INFO.WIN_HEIGHT),
            min(self.head.x, GLOBAL_INFO.WIN_HEIGHT - self.head.y)
            * GLOBAL_INFO.ROOT2
            / GLOBAL_INFO.WIN_DIAGONAL,
            self.head.x / GLOBAL_INFO.WIN_WIDTH,
            min(self.head.x, self.head.y)
            * GLOBAL_INFO.ROOT2
            / GLOBAL_INFO.WIN_DIAGONAL,
        ]

        # 4. One-Hot Encoded Direction of Snake (One-Hot Encoded) (V4 Directions)
        snake_direction = [
            self.direction == Direction.RIGHT,
            self.direction == Direction.DOWN,
            self.direction == Direction.LEFT,
            self.direction == Direction.UP,
        ]

        # 5. Direction of Tail (One-Hot Encoded) (V4 Directions)
        tail = self.snake[-1]
        body = self.snake[-2]
        tail_direction = [
            (tail.y == body.y) and (tail.x < body.x),
            (tail.x == body.x) and (tail.y < body.y),
            (tail.y == body.y) and (tail.x > body.x),
            (tail.x == body.x) and (tail.y > body.y),
        ]

        return tuple(
            food_inputs + snake_body + walls + snake_direction + tail_direction
        )


# Gene evaluation function for NEAT
def eval_genome(genomes, config):
    individual = 0

    for gene_id, gene in genomes:
        individual += 1
        # Set initial fitness of all genes = 0, create neural networks and add then to the lists above
        gene.fitness = 0
        neural_net = neat.nn.FeedForwardNetwork.create(gene, config)

        snake = Snake(GLOBAL_INFO.GEN, individual)

        run = True
        while run:
            outputs = neural_net.activate(snake.get_Vision())
            action = outputs.index(max(outputs))
            value = snake.play_step(action)

            if len(snake.snake) > GLOBAL_INFO.AVG_LENGTH:
                GLOBAL_INFO.SPEED = 40
            else:
                GLOBAL_INFO.SPEED = 120

            if value[0]:
                run = False
                gene.fitness = value[1] + (value[2] / 1000)

        GLOBAL_INFO.AVG_LENGTH = (
            GLOBAL_INFO.AVG_LENGTH * GLOBAL_INFO.NO_SNAKES + len(snake.snake)
        ) / (GLOBAL_INFO.NO_SNAKES + 1)
        GLOBAL_INFO.NO_SNAKES += 1
    GLOBAL_INFO.GEN += 1


def run(config_file):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genome, 100)

    print("\nBest genome:\n{!s}".format(winner))


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config_file.txt")
    run(config_path)
