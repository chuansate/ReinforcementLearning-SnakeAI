class QSnake:
    def __init__(self, SPACE_SIZE):
        self.coordinates = []  # coordinates of the snake body (self.coordinates[0] is the snake head)
        self.coordinates.append((0, 0))
        self.coordinates.append((SPACE_SIZE, SPACE_SIZE))
