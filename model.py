from snakeImageVision import Snake
import numpy as np
import cv2
import os
import torch
from torch import nn
import pandas as pd
import torch.optim as optim
from collections import deque
import random


# game = Snake()

# example = [0, 0, 0, 0, 1]


# for i in example:
#     game.play_step(i)

# image = cv2.imread("ImageVision.jpg", cv2.IMREAD_GRAYSCALE)
# image = cv2.resize(image, (64, 64))

# cv2.imwrite("grayscale.jpeg", image)

# ret, thresh1 = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)

# image_csv = pd.DataFrame(thresh1)

# image_csv.to_csv("Image6464")

# cv2.imwrite("binary_image.jpeg", thresh1)

# image_tensor = torch.tensor(thresh1)

# print(image_tensor)

# print(image_tensor.shape)


class CQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convulutions = nn.Sequential(
            nn.Conv2d(1, 16, padding="same", kernel_size=3),  # Out = 64*64*16
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Out = 16*32*32
            nn.Conv2d(16, 16, padding="same", kernel_size=3),  # Out = 32*32*16
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Out = 16*16*16
            nn.Conv2d(16, 8, padding="same", kernel_size=3),  # Out = 16*16*8
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Out = 8*8*8
            nn.Flatten(0, -1),  # 512
        )
        self.RL = nn.Sequential(
            nn.Linear(512, 100),  # 100
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 3),
        )

    def forward(self, x):
        # x is a 64*64 image
        x = self.convulutions(x)
        x = self.RL(x)
        return x

    def save(self, file_name="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)

        next_state = torch.tensor(next_state, dtype=torch.float)

        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 3:
            # state = torch.unsqueeze(state, 0)
            # next_state = torch.unsqueeze(next_state, 0)
            # action = torch.unsqueeze(action, 0)
            # reward = torch.unsqueeze(reward, 0)
            # done = (done,)
            predicted = self.model(state)

            target = predicted.clone()

            Q_new = reward

            if not done:
                Q_new += self.gamma * torch.max(self.model(next_state))

            target[torch.argmax(action).item()] = Q_new

        else:

            predicted = torch.zeros(len(done), 3)
            target = predicted.clone()

            for idx in range(len(done)):
                predicted[idx] = self.model(state[idx])

                target[idx] = predicted[idx].clone()

                Q_new = reward[idx]

                if not done[idx]:
                    Q_new += self.gamma * torch.max(self.model(next_state[idx]))

                target[idx][torch.argmax(action).item()] = Q_new

        # predicted = self.model(state)
        # predicted = predicted.view(-1, 3)

        # target = predicted.clone()

        # for idx in range(len(done)):
        #     Q_new = reward[idx]

        #     if not done[idx]:
        #         Q_new += self.gamma * torch.max(self.model(next_state[idx]))

        #     target[idx][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, predicted)

        loss.backward()

        self.optimizer.step()


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = CQN()
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self):
        image = cv2.imread("ImageVision.jpg", cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (64, 64))
        ret, bin_image = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
        bin_image = bin_image.reshape((1, bin_image.shape[0], bin_image.shape[1]))
        return bin_image

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
        self.epsilon = 190 - self.n_games
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
    record = 0

    agent = Agent()

    game = Snake()

    while True:
        state_old = agent.get_state()

        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)

        state_new = agent.get_state()

        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print("Game", agent.n_games, "Score", score, "Record", record)


if __name__ == "__main__":
    train()
