from tkinter import *
import random
from QSnake import QSnake
from QFood import QFood
import numpy as np


WIDTH = 500
HEIGHT = 500
SPEED = 1000  # the snake moves after SPEED in milliseconds
SPACE_SIZE = 20
BODY_SIZE = 2
SNAKE = "green"
SNAKE_HEAD = "yellow"
FOOD = "white"
BACKGROUND = "black"
MOVE_PENALTY = 1
COLLISION_PENALTY = 300
FOOD_REWARD = 500


class QLearningEnv:
    score = 0
    direction = "down"  # direction of inertia of the snake

    def __init__(self):
        self.window_width = WIDTH
        # height of window, altho we set it as 500, but it will be higher than that becoz we got extra bar
        self.window_height = HEIGHT

        self.snake = QSnake(SPACE_SIZE)
        self.food = QFood(self.snake, WIDTH, SPACE_SIZE, HEIGHT, FOOD)
        self.action_space = {
            0: 'left',
            1: 'right',
            2: 'up',
            3: 'down'
        }
        self.board = np.zeros((HEIGHT // SPACE_SIZE, WIDTH // SPACE_SIZE))
        self.board[0][0] = 1
        self.board[1][0] = 1
        self.board[self.food.coordinates[1] // SPACE_SIZE][self.food.coordinates[0] // SPACE_SIZE] = 2

    #  render using cv2
    def render(self, speed_snake):
        pass

    def check_collisions(self):
        x, y = self.snake.coordinates[0]
        if x < 0 or x >= WIDTH:
            return True
        elif y < 0 or y >= HEIGHT:
            print("Collision happens!")
            print("y = ", y)
            return True

        for body_part in self.snake.coordinates[1:]:
            if x == body_part[0] and y == body_part[1]:
                return True

        return False

    def reset(self):
        QLearningEnv.score = 0
        QLearningEnv.direction = "down"
        self.window_width = WIDTH
        # height of window, altho we set it as 500, but it will be higher than that becoz we got extra bar
        self.window_height = HEIGHT

        self.snake = QSnake(SPACE_SIZE)
        self.food = QFood(self.snake, WIDTH, SPACE_SIZE, HEIGHT, FOOD)
        self.board = np.zeros((HEIGHT // SPACE_SIZE, WIDTH // SPACE_SIZE))
        self.board[0][0] = 1
        self.board[1][0] = 1
        self.board[self.food.coordinates[1]//SPACE_SIZE][self.food.coordinates[0]//SPACE_SIZE] = 2

        return self.get_state()

    def step(self, action):
        new_state = 0
        reward = 0
        done = False

        if action == 0 and QLearningEnv.direction != "right":
            QLearningEnv.direction = "left"
        elif action == 1 and QLearningEnv.direction != "left":
            QLearningEnv.direction = "right"
        elif action == 2 and QLearningEnv.direction != "down":
            QLearningEnv.direction = "up"
        elif action == 3 and QLearningEnv.direction != "up":
            QLearningEnv.direction = "down"

        x, y = self.snake.coordinates[0]
        if QLearningEnv.direction == "up":
            reward -= MOVE_PENALTY
            y -= SPACE_SIZE
        elif QLearningEnv.direction == "down":
            reward -= MOVE_PENALTY
            y += SPACE_SIZE
        elif QLearningEnv.direction == "left":
            reward -= MOVE_PENALTY
            x -= SPACE_SIZE
        elif QLearningEnv.direction == "right":
            reward -= MOVE_PENALTY
            x += SPACE_SIZE

        new_state = self.get_state()
        self.snake.coordinates.insert(0, (x, y))

        if self.check_collisions():
            done = True
            reward -= COLLISION_PENALTY

            return new_state, reward, done

        self.board[y // SPACE_SIZE][x // SPACE_SIZE] = 1

        if x == self.food.coordinates[0] and y == self.food.coordinates[1]:
            reward += FOOD_REWARD
            QLearningEnv.score += 1
            self.board[self.food.coordinates[1]//SPACE_SIZE][self.food.coordinates[0]//SPACE_SIZE] = 0
            self.food = QFood(self.snake, WIDTH, SPACE_SIZE, HEIGHT, FOOD)
            self.board[self.food.coordinates[1] // SPACE_SIZE][self.food.coordinates[0] // SPACE_SIZE] = 2
        else:
            tail_x = self.snake.coordinates[-1][0]
            tail_y = self.snake.coordinates[-1][1]
            self.board[tail_y // SPACE_SIZE][tail_x // SPACE_SIZE] = 0
            del self.snake.coordinates[-1]

        new_state = self.get_state()
        return new_state, reward, done

    def get_state(self):
        state = []
        snake_head = self.snake.coordinates[0]
        dist_x = self.food.coordinates[0] - snake_head[0]
        dist_y = self.food.coordinates[1] - snake_head[1]
        state.append(int(self.direction == "left"))
        state.append(int(self.direction == "right"))
        state.append(int(self.direction == "up"))
        state.append(int(self.direction == "down"))
        state.append(int(dist_x < 0))  # food is at the left of snake
        state.append(int(dist_x > 0))  # food is at the right of snake
        state.append(int(dist_y < 0))  # food is up
        state.append(int(dist_y > 0))  # food is down
        state.append(int(self.danger_is_left()))  # danger is left (body is at left, or wall is at left)
        state.append(int(self.danger_is_right()))
        state.append(int(self.danger_is_up()))
        state.append(int(self.danger_is_down()))

        return tuple(state)

    def danger_is_left(self):
        head_x_index = self.snake.coordinates[0][0]//SPACE_SIZE
        head_y_index = self.snake.coordinates[0][1]//SPACE_SIZE
        if self.direction == "right":
            return False

        if head_x_index - 1 < 0:  # wall is at left
            return True

        if self.board[head_y_index][head_x_index - 1] == 1:  # body is at left
            return True

        return False

    def danger_is_right(self):
        head_x_index = self.snake.coordinates[0][0]//SPACE_SIZE
        head_y_index = self.snake.coordinates[0][1]//SPACE_SIZE
        if self.direction == "left":
            return False

        if head_x_index + 1 == WIDTH//SPACE_SIZE:  # wall is at right
            return True

        if self.board[head_y_index][head_x_index + 1] == 1:  # body is at right
            return True

        return False

    def danger_is_up(self):
        head_x_index = self.snake.coordinates[0][0] // SPACE_SIZE
        head_y_index = self.snake.coordinates[0][1] // SPACE_SIZE
        if self.direction == "down":
            return False

        if head_y_index - 1 < 0:  # wall is at up
            return True

        if self.board[head_y_index - 1][head_x_index] == 1:  # body is at up
            return True

        return False

    def danger_is_down(self):
        head_x_index = self.snake.coordinates[0][0]//SPACE_SIZE
        head_y_index = self.snake.coordinates[0][1]//SPACE_SIZE
        if self.direction == "up":
            return False

        if head_y_index + 1 == WIDTH//SPACE_SIZE:  # wall is at down
            return True

        if self.board[head_y_index + 1][head_x_index] == 1:  # body is at down
            return True

        return False
