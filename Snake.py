SNAKE_HEAD = "yellow"


class Snake:
    def __init__(self, canvas, BODY_SIZE, SPACE_SIZE, SNAKE):
        self.body_size = BODY_SIZE
        self.coordinates = []  # coordinates of the snake body (self.coordinates[0] is the snake head)
        self.squares = []  # canvas components//shapes of the snake body

        for i in range(0, BODY_SIZE):
            self.coordinates.append([0, 0])

        for x, y in self.coordinates:
            square = canvas.create_rectangle(x, y, x + SPACE_SIZE, y + SPACE_SIZE, fill=SNAKE, tag="snake")
            self.squares.append(square)







