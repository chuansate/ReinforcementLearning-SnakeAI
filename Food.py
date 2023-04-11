import random


class Food:
    def __init__(self, canvas, snake, WIDTH, SPACE_SIZE, HEIGHT, FOOD):
        x = random.randint(0, (WIDTH / SPACE_SIZE) - 1) * SPACE_SIZE
        y = random.randint(0, (HEIGHT / SPACE_SIZE) - 1) * SPACE_SIZE
        while True:
            if (x, y) in snake.coordinates:
                x = random.randint(0, (WIDTH / SPACE_SIZE) - 1) * SPACE_SIZE
                y = random.randint(0, (HEIGHT / SPACE_SIZE) - 1) * SPACE_SIZE
            else:
                break

        self.coordinates = [x, y]

        canvas.create_oval(x, y, x + SPACE_SIZE, y +
                           SPACE_SIZE, fill=FOOD, tag="food")