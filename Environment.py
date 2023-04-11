from tkinter import *
import random
from Snake import Snake
from Food import Food
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


# implement the get_state, apply action on env...
class Environment:
    score = 0
    direction = "down"  # direction of inertia of the snake

    def __init__(self):
        self.window = Tk()
        self.window.title("Snake Game")
        self.label = Label(self.window, text="Points:{}".format(Environment.score),
                      font=('consolas', 20))
        self.label.pack()

        self.canvas = Canvas(self.window, bg=BACKGROUND,
                        height=HEIGHT, width=WIDTH)
        self.canvas.pack()

        self.window.update()

        window_width = self.window.winfo_width()
        # height of window, altho we set it as 500, but it will be higher than that becoz we got extra bar
        window_height = self.window.winfo_height()
        screen_width = self.window.winfo_screenwidth()  # width of computer screen
        screen_height = self.window.winfo_screenheight()  # height of computer screen

        x = int((screen_width / 2) - (window_width / 2))
        y = int((screen_height / 2) - (window_height / 2))

        self.window.geometry(f"{window_width}x{window_height}+{x}+{y}")

        self.window.bind('<Left>',
                    lambda event: self.change_direction('left'))
        self.window.bind('<Right>',
                    lambda event: self.change_direction('right'))
        self.window.bind('<Up>',
                    lambda event: self.change_direction('up'))
        self.window.bind('<Down>',
                    lambda event: self.change_direction('down'))

        self.snake = Snake(self.canvas, BODY_SIZE, SPACE_SIZE, SNAKE)
        self.food = Food(self.canvas, self.snake, WIDTH, SPACE_SIZE, HEIGHT, FOOD)

        self.next_turn()

        self.window.mainloop()

    # Function to check the next move of snake
    def next_turn(self):
        x, y = self.snake.coordinates[0]
        if Environment.direction == "up":
            y -= SPACE_SIZE
        elif Environment.direction == "down":
            y += SPACE_SIZE
        elif Environment.direction == "left":
            x -= SPACE_SIZE
        elif Environment.direction == "right":
            x += SPACE_SIZE

        self.snake.coordinates.insert(0, (x, y))

        square = self.canvas.create_rectangle(
            x, y, x + SPACE_SIZE,
                  y + SPACE_SIZE, fill=SNAKE_HEAD)  # id (type int) of the canvas components//shapes

        self.snake.squares.insert(0, square)
        for i in range(1, len(self.snake.coordinates) - 1):
            body_x = self.snake.coordinates[i][0]
            body_y = self.snake.coordinates[i][1]
            self.canvas.delete(self.snake.squares[i])
            square = self.canvas.create_rectangle(
                body_x, body_y, body_x + SPACE_SIZE,
                                body_y + SPACE_SIZE, fill=SNAKE)
            self.snake.squares[i] = square

        # if snake eats food
        if x == self.food.coordinates[0] and y == self.food.coordinates[1]:
            Environment.score += 1
            self.label.config(text="Points:{}".format(Environment.score))
            self.canvas.delete("food")
            self.food = Food(self.canvas, self.snake, WIDTH, SPACE_SIZE, HEIGHT, FOOD)

        else:
            tail_x = self.snake.coordinates[-1][0]
            tail_y = self.snake.coordinates[-1][1]
            del self.snake.coordinates[-1]
            self.canvas.delete(self.snake.squares[-1])
            del self.snake.squares[-1]

        if self.check_collisions():
            self.game_over()
        else:
            self.window.after(SPEED, self.next_turn)  # recursive call of `next_turn` after certain period

    # Function to control direction of snake
    def change_direction(self, new_direction):
        if new_direction == 'left':
            if Environment.direction != 'right':
                Environment.direction = new_direction
        elif new_direction == 'right':
            if Environment.direction != 'left':
                Environment.direction = new_direction
        elif new_direction == 'up':
            if Environment.direction != 'down':
                Environment.direction = new_direction
        elif new_direction == 'down':
            if Environment.direction != 'up':
                Environment.direction = new_direction

    # function to check snake's collision and position
    def check_collisions(self):
        x, y = self.snake.coordinates[0]

        if x < 0 or x >= WIDTH:
            return True
        elif y < 0 or y >= HEIGHT:
            return True

        for body_part in self.snake.coordinates[1:]:
            if x == body_part[0] and y == body_part[1]:
                return True

        return False

    # Function to control everything
    def game_over(self):
        self.canvas.delete(ALL)
        self.canvas.create_text(self.canvas.winfo_width() / 2,
                           self.canvas.winfo_height() / 2,
                           font=('consolas', 70),
                           text="GAME OVER", fill="red",
                           tag="gameover")

    def reset(self):
        return self.get_state()

    def step(self, action):
        new_state = 0
        reward = 0
        done = True
        info = True
        return new_state, reward, done


