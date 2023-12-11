import torch
import random
import numpy as np
from collections import deque
from snake_game_AI import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
BLOCK_SIZE = 20


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9  # dicount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(14, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):

        def point_r(pt, dist):
            return Point(pt.x + BLOCK_SIZE*dist, pt.y)

        def point_l(pt, dist):
            return Point(pt.x-BLOCK_SIZE*dist, pt.y)

        def point_u(pt, dist):
            return Point(pt.x, pt.y - BLOCK_SIZE*dist)

        def point_d(pt, dist):
            return Point(pt.x, pt.y + BLOCK_SIZE*dist)

        head = game.snake[0]

        clockwise_direction = [Direction.RIGHT, Direction.DOWN,
                               Direction.LEFT, Direction.UP]
        idx = clockwise_direction.index(game.direction)

        straight = idx
        right = (idx+1) % 4
        left = (idx-1) % 4

        clockwise_points = [point_r, point_d, point_l, point_u]

        distance = 1
        while not game.is_wall(clockwise_points[straight](head, distance)):
            distance += 1
        wall_straight = distance

        distance = 1
        while not game.is_wall(clockwise_points[right](head, distance)):
            distance += 1
        wall_right = distance

        distance = 1
        while not game.is_wall(clockwise_points[left](head, distance)):
            distance += 1
        wall_left = distance

        distance = 1
        while not game.is_body(clockwise_points[straight](head, distance)):
            distance += 1
            if game.is_wall(clockwise_points[straight](head, distance)):
                break
        body_straight = distance

        if body_straight == wall_straight:
            body_straight = 0

        distance = 1
        while not game.is_body(clockwise_points[right](head, distance)):
            distance += 1
            if game.is_wall(clockwise_points[right](head, distance)):
                break
        body_right = distance

        if body_right == wall_right:
            body_right = 0

        distance = 1
        while not game.is_body(clockwise_points[left](head, distance)):
            distance += 1
            if game.is_wall(clockwise_points[left](head, distance)):
                break
        body_left = distance

        if body_left == wall_left:
            body_left = 0

        # danger_straight = game.is_collision(
        #     clockwise_points[straight](head, 1))
        # danger_right = game.is_collision(clockwise_points[right](head, 1))
        # danger_left = game.is_collision(clockwise_points[left](head, 1))

        # danger_straight_2 = game.is_collision(
        #     clockwise_points[straight](head, 2))
        # danger_right_2 = game.is_collision(clockwise_points[right](head, 2))
        # danger_left_2 = game.is_collision(clockwise_points[left](head, 2))

        # danger_straight_right = game.is_collision(
        #     clockwise_points[right](clockwise_points[straight](head, 1), 1))
        # danger_straight_left = game.is_collision(
        #     clockwise_points[left](clockwise_points[straight](head, 1), 1))

        dir_r = game.direction == Direction.RIGHT
        dir_l = game.direction == Direction.LEFT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [

            #Distance Wall
            wall_straight,
            wall_right,
            wall_left,

            #Distance Body
            body_straight,
            body_right,
            body_left,

            # # Danger 1
            # danger_straight,
            # danger_right,
            # danger_left,

            # move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # food location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    board_size = 10
    w = board_size*BLOCK_SIZE
    h = board_size*BLOCK_SIZE
    game = SnakeGameAI(w,h)

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(
            state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
