# from snakeImageVision import Snake, Direction, Point
import numpy as np
import cv2
import os
import torch
from torch import nn

# game = Snake()

# example = [0, 0, 0, 0, 1]


# for i in example:
#     game.play_step(i)

image = cv2.imread("ImageVision.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (64, 64))

cv2.imwrite("grayscale.jpeg", image)

ret, thresh1 = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)

cv2.imwrite("binary_image.jpeg", thresh1)

image_tensor = torch.tensor(thresh1)

print(image_tensor)

print(image_tensor.shape)


class CQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convulutions = nn.Sequential(
            nn.Conv2d(1, 16, padding="same", kernel_size=3),  # Out = 16*64*64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Out = 16*32*32
            nn.Conv2d(16, 16, padding="same", kernel_size=3), # Out = 16*32*32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Out = 16*16*16
            nn.Conv2d(16, 8, padding="same", kernel_size=3),  # Out =
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
