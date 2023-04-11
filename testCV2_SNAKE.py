import cv2
import numpy as np
from QLearningEnv import QLearningEnv

SPACE_SIZE = 20

env = QLearningEnv()
actions = [3, 3, 3, 1, 3, 3, 3, 3]  # for testing purposes
food_positions = [(0, 40), (0, 80), (0, 80), (0, 200), (0, 200), (0, 200), (0, 200), (0, 200)]
print("Agent made to the final timestep!")
snake_body_coordinates = [(0, SPACE_SIZE), (0, 0)]
# actions_names = ["left", "right", "up", "down"]
direction = "down"
timestep_index = 0
snake_scores = 0
img = np.zeros((env.window_width, env.window_height, 3), dtype=np.uint8)
head_x = snake_body_coordinates[0][0]
head_y = snake_body_coordinates[0][1]
cv2.rectangle(img, (0, 0), (0 + SPACE_SIZE, 0 + SPACE_SIZE), (0, 128, 0), -1)  # SNAKE_HEAD = yellow, snake_body = green, food = white
cv2.rectangle(img, (head_x, head_y), (head_x + SPACE_SIZE, head_y + SPACE_SIZE), (0, 255, 255), -1)
cv2.circle(img, (food_positions[timestep_index][0] + SPACE_SIZE//2, food_positions[timestep_index][1] + SPACE_SIZE//2), SPACE_SIZE//2, (255, 255, 255), -1)
print("Scores = ", snake_scores)
while True:
    cv2.imshow("Snake AI Gameplay", img)
    """0: 'left',
    1: 'right',
    2: 'up',
    3: 'down'"""
    if actions[timestep_index] == 0:
        head_x -= SPACE_SIZE
    elif actions[timestep_index] == 1:
        head_x += SPACE_SIZE
    elif actions[timestep_index] == 2:
        head_y -= SPACE_SIZE
    elif actions[timestep_index] == 3:
        head_y += SPACE_SIZE

    snake_body_coordinates.insert(0, (head_x, head_y))
    timestep_index += 1  # snake head is moving in intended direction, means going to next timestep
    print("time_step_index = ", timestep_index)
    print("snake_body_coordinates = ", snake_body_coordinates)
    if snake_body_coordinates[0][0] == food_positions[timestep_index-1][0] and snake_body_coordinates[0][1] == food_positions[timestep_index-1][1]:############ MODIFIED!
        snake_scores += 1
        print("Scores = ", snake_scores)
        ############################################################################ After eating the food, new food won't be spawned
        ##################################################### try to represent states by including 2 more states:
        # vertical distance of head to food : 0 to 24
        # horizontal distance of head to food : 0 to 24
        # 2^12 * 25 * 25 = 2560000
        # print("food_positions[timestep_index][0]", food_positions[timestep_index][0])
        # print("food_positions[timestep_index+1][0]", food_positions[timestep_index+1][0])
        cv2.circle(img, (food_positions[timestep_index+1][0] + SPACE_SIZE // 2,
                         food_positions[timestep_index+1][1] + SPACE_SIZE // 2), SPACE_SIZE // 2,
                   (255, 255, 255), -1)  # update the food
        # if the food is eaten, the tail won't be popped
        head = True
        for body_x, body_y in snake_body_coordinates:
            if head:
                cv2.rectangle(img, (body_x, body_y), (body_x+SPACE_SIZE, body_y+SPACE_SIZE), (0, 255, 255), -1)
                head = False
            else:
                cv2.rectangle(img, (body_x, body_y), (body_x+SPACE_SIZE, body_y+SPACE_SIZE), (0, 128, 0), -1)

    else:
        # move the tail by painting previous position as black
        cv2.rectangle(img, (snake_body_coordinates[-1][0], snake_body_coordinates[-1][1]), (snake_body_coordinates[-1][0] + SPACE_SIZE, snake_body_coordinates[-1][1] + SPACE_SIZE), (0, 0, 0), -1)
        snake_body_coordinates.pop()
        head = True
        for body_x, body_y in snake_body_coordinates:
            if head:
                cv2.rectangle(img, (body_x, body_y), (body_x+SPACE_SIZE, body_y+SPACE_SIZE), (0, 255, 255), -1)
                head = False
            else:
                cv2.rectangle(img, (body_x, body_y), (body_x+SPACE_SIZE, body_y+SPACE_SIZE), (0, 128, 0), -1)

    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()