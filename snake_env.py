import pygame
import numpy as np
import gym
from gym import spaces
import random

class SnakeEnv(gym.Env):
    def __init__(self):
        super(SnakeEnv, self).__init__()
        self.width = 800
        self.height = 600
        self.snake_block = 20
        self.snake_speed = 50

        self.action_space = spaces.Discrete(4)  # 0: left, 1: right, 2: up, 3: down
        self.observation_space = spaces.Box(low=0, high=max(self.width, self.height), shape=(11,), dtype=np.float32)

        pygame.init()
        self.dis = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')

        self.clock = pygame.time.Clock()

    def reset(self):
        self.x1 = self.width / 2
        self.y1 = self.height / 2
        self.x1_change = 0
        self.y1_change = 0
        self.snake_List = [[self.x1, self.y1]]
        self.Length_of_snake = 1
        self.foodx = round(random.randrange(0, self.width - self.snake_block) / 20.0) * 20.0
        self.foody = round(random.randrange(0, self.height - self.snake_block) / 20.0) * 20.0
        self.score = 0
        self.done = False

        state = self.get_state()
        return np.array(state, dtype=np.float32)

    def step(self, action):
        if action == 0:
            self.x1_change = -self.snake_block
            self.y1_change = 0
        elif action == 1:
            self.x1_change = self.snake_block
            self.y1_change = 0
        elif action == 2:
            self.x1_change = 0
            self.y1_change = -self.snake_block
        elif action == 3:
            self.x1_change = 0
            self.y1_change = self.snake_block

        self.x1 += self.x1_change
        self.y1 += self.y1_change
        self.snake_List.append([self.x1, self.y1])

        if len(self.snake_List) > self.Length_of_snake:
            del self.snake_List[0]

        reward = 0
        if self.x1 == self.foodx and self.y1 == self.foody:
            self.foodx = round(random.randrange(0, self.width - self.snake_block) / 20.0) * 20.0
            self.foody = round(random.randrange(0, self.height - self.snake_block) / 20.0) * 20.0
            self.Length_of_snake += 1
            self.score += 1
            reward = 10

        if self.x1 >= self.width or self.x1 < 0 or self.y1 >= self.height or self.y1 < 0:
            self.done = True
            reward = -10
        elif [self.x1, self.y1] in self.snake_List[:-1]:
            self.done = True
            reward = -10
        else:
            reward = 1 if (abs(self.foodx - self.x1) + abs(self.foody - self.y1)) < (abs(self.foodx - (self.x1 - self.x1_change)) + abs(self.foody - (self.y1 - self.y1_change))) else -1

        state = self.get_state()
        return np.array(state, dtype=np.float32), reward, self.done, {}

    def get_state(self):
        head = self.snake_List[-1]
        point_l = [head[0] - self.snake_block, head[1]]
        point_r = [head[0] + self.snake_block, head[1]]
        point_u = [head[0], head[1] - self.snake_block]
        point_d = [head[0], head[1] + self.snake_block]

        dir_l = self.x1_change == -self.snake_block
        dir_r = self.x1_change == self.snake_block
        dir_u = self.y1_change == -self.snake_block
        dir_d = self.y1_change == self.snake_block

        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or 
            (dir_l and self.is_collision(point_l)) or 
            (dir_u and self.is_collision(point_u)) or 
            (dir_d and self.is_collision(point_d)),

            # Danger right
            (dir_u and self.is_collision(point_r)) or 
            (dir_d and self.is_collision(point_l)) or 
            (dir_l and self.is_collision(point_u)) or 
            (dir_r and self.is_collision(point_d)),

            # Danger left
            (dir_d and self.is_collision(point_r)) or 
            (dir_u and self.is_collision(point_l)) or 
            (dir_r and self.is_collision(point_u)) or 
            (dir_l and self.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            self.foodx < head[0],  # food left
            self.foodx > head[0],  # food right
            self.foody < head[1],  # food up
            self.foody > head[1]   # food down
        ]

        return state

    def is_collision(self, point):
        return point[0] >= self.width or point[0] < 0 or point[1] >= self.height or point[1] < 0 or point in self.snake_List

    def render(self, mode='human'):
        self.dis.fill((0, 0, 0))

        for segment in self.snake_List:
            pygame.draw.rect(self.dis, (0, 255, 0), [segment[0], segment[1], self.snake_block, self.snake_block])

        pygame.draw.rect(self.dis, (255, 0, 0), [self.foodx, self.foody, self.snake_block, self.snake_block])

        pygame.display.update()
        self.clock.tick(self.snake_speed)

    def close(self):
        pygame.quit()
