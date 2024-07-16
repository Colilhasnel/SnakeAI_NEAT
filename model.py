from snakeImageVision import Snake
import numpy as np
import cv2
import os
import torch
from torch import nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from IPython import display

losses = []


class CQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convulutions = nn.Sequential(
            # Input = 4*128*128 (Channel, Height, Width)
            nn.Conv2d(
                in_channels=4, out_channels=32, kernel_size=8, stride=4
            ),  # Out = 32*31*31
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Out = 32*15*15
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2
            ),  # Out = 64*6*6
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Out = 64*3*3
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=2
            ),  # Out = 128*1*1
            nn.ReLU(),
            nn.Flatten(),  # Out = 128*1
        )
        self.RL = nn.Sequential(
            nn.Linear(128, 512),  # 512
            nn.ReLU(),
            nn.Linear(512, 512),  # 512
            nn.ReLU(),
            nn.Linear(512, 3),  # 3
        )

    def forward(self, x):
        # x is a 128*128 image
        x = self.convulutions(x)  # Out = 128*1
        x = x.view(-1, 128)  # Out = 1*128
        x = self.RL(x)  # = 1*3
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
        self.current_loss = 0
        self.moves = 1

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)

        next_state = torch.tensor(next_state, dtype=torch.float)

        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 3:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        predicted = self.model(state)

        target = predicted.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]

            if not done[idx]:
                Q_new += self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, predicted)

        if len(done) > 1:
            self.current_loss = self.current_loss / self.moves
            losses.append(self.current_loss)
            self.current_loss = 0
            self.moves = 1

        else:
            self.current_loss += loss.item()
            self.moves += 1

        loss.backward()

        self.optimizer.step()


MAX_MEMORY = 100_000
BATCH_SIZE = 300
LR = 0.01


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = CQN()
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, frame_iteration):

        images = [
            cv2.imread("ImageVision1.jpg", cv2.IMREAD_GRAYSCALE),
            cv2.imread("ImageVision2.jpg", cv2.IMREAD_GRAYSCALE),
            cv2.imread("ImageVision3.jpg", cv2.IMREAD_GRAYSCALE),
            cv2.imread("ImageVision4.jpg", cv2.IMREAD_GRAYSCALE),
        ]

        bin_images = [[], [], [], []]

        for i in range(0, 4):
            images[i] = cv2.resize(images[i], (128, 128))
            ret, images[i] = cv2.threshold(images[i], 10, 255, cv2.THRESH_BINARY)

        frame_arrange = frame_iteration % 4

        for i in range(0, 4):
            bin_images[i] = images[(frame_arrange - i + 4) % 4]

        # image1 = cv2.imread("ImageVision.jpg", cv2.IMREAD_GRAYSCALE)
        # image1 = cv2.resize(image1, (128, 128))
        # ret, bin_image = cv2.threshold(image1, 10, 255, cv2.THRESH_BINARY)
        # bin_image = bin_image.reshape((1, bin_image.shape[0], bin_image.shape[1]))
        return bin_images

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
        self.epsilon = 300 - self.n_games
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

    plt.ion()

    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title("Training")
    plt.xlabel("Number of Games")
    plt.ylabel("Loss")

    agent = Agent()

    game = Snake()

    while True:
        state_old = agent.get_state(game.frame_iteration)

        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)

        state_new = agent.get_state(game.frame_iteration)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            plt.plot(losses)

            plt.pause(0.1)

            print("Game", agent.n_games, "Score", score, "Record", record)


if __name__ == "__main__":
    train()
