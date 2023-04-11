import random
import numpy as np
from QLearningEnv import QLearningEnv
import pickle
import cv2
import matplotlib.pyplot as plt

NUM_EPISODE = 20000
NUM_TIMESTEP = 1000
epsilon = 0.8
EPS_DECAY = 0.9999  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 1000  # how often to play through env visually.
learning_rate = 0.1
discount_rate = 0.99
start_q_table = None  # if we have a picked Q table, put its filename here
SPACE_SIZE = 20
# num_cols = 25
# num_rows = 25


def arg_max(li, skipped_action_index):
    li = li.tolist()
    max_val = -999
    max_index = -1
    for k in range(0, len(li)):
        if k == skipped_action_index:
            continue

        if li[k] > max_val:
            max_val = li[k]
            max_index = k
    return max_index


"""q_table = {}
for i in range(0, 2):
    for ii in range(0, 2):
        for iii in range(0, 2):
            for iiii in range(0, 2):
                for iiiii in range(0, 2):
                    for iiiiii in range(0, 2):
                        for j in range(0, 2):
                            for jj in range(0, 2):
                                for jjj in range(0, 2):
                                    for jjjj in range(0, 2):
                                        for jjjjj in range(0, 2):
                                            for jjjjjj in range(0, 2):
                                                # keys are just nested tuples
                                                q_table[(i, ii, iii, iiii, iiiii, iiiiii, j, jj, jjj, jjjj, jjjjj, jjjjjj)] = np.array([0 for i in
                                                                                   range(4)])"""

if __name__ == "__main__":
    if start_q_table is None:
        # initialize the q-table#
        q_table = dict()
        for i in range(0, 2):
            for ii in range(0, 2):
                for iii in range(0, 2):
                    for iiii in range(0, 2):
                        for iiiii in range(0, 2):
                            for iiiiii in range(0, 2):
                                for j in range(0, 2):
                                    for jj in range(0, 2):
                                        for jjj in range(0, 2):
                                            for jjjj in range(0, 2):
                                                for jjjjj in range(0, 2):
                                                    for jjjjjj in range(0, 2):
                                                        # keys are just nested tuples
                                                        q_table[(i, ii, iii, iiii, iiiii, iiiiii, j, jj, jjj, jjjj, jjjjj, jjjjjj)] = np.array([0 for i in
                                                                                           range(4)], dtype="float32")  # recall the 4 movements by blob
    else:
        with open(start_q_table, "rb") as f:
            q_table = pickle.load(f)

    """print(q_table[(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)])
    print(type(q_table[(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]))
    exit()"""
    episode_rewards = []
    episode_points = []
    actions = []
    food_positions = []
    for episode in range(1, NUM_EPISODE+1):
        #print(f"Running Episode {str(episode)}....")
        env = QLearningEnv()
        state = env.reset()
        done = False
        rewards_current_episode = 0

        # for every 1000 episodes, show its mean rewards
        if episode % SHOW_EVERY == 0:
            print(f"On Episode {episode}, epsilon is {epsilon}")
            print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
            # show = True   `show` is for visualizing the gameplay every x episodes
        # else:
            # show = False

        for timestep in range(0, NUM_TIMESTEP):
            exploration_rate_threshold = random.uniform(0, 1)
            if exploration_rate_threshold > epsilon:
                # agent exploits the env and choose the action that has the highest Q-value
                # in the Q-table for the current state.
                #print("Agent exploiting env....")
                if QLearningEnv.direction == "left":
                    action = arg_max(q_table[state], 1)
                elif QLearningEnv.direction == "right":
                    action = arg_max(q_table[state], 0)
                elif QLearningEnv.direction == "up":
                    action = arg_max(q_table[state], 3)
                elif QLearningEnv.direction == "down":
                    """if episode > NUM_EPISODE - 9:
                        print("direction is down...")"""
                    action = arg_max(q_table[state], 2)
                # action = np.argmax(q_table[state])

            else:
                """self.action_space = {
                    0: 'left',
                    1: 'right',
                    2: 'up',
                    3: 'down'
                }"""
                #print("Agent exploring env....")
                # explore the env by randomly sampling an action
                if QLearningEnv.direction == "left":
                    action = random.sample([0, 2, 3], 1)
                elif QLearningEnv.direction == "right":
                    action = random.sample([1, 2, 3], 1)
                elif QLearningEnv.direction == "up":
                    action = random.sample([1, 2, 0], 1)
                elif QLearningEnv.direction == "down":
                    action = random.sample([1, 0, 3], 1)
                action = action[0]

            actions.append(action)
            food_positions.append(env.food.coordinates)
            #if episode > NUM_EPISODE - 9:
            #print("state = ", state)
            #print("q_table[state] = ", q_table[state]) #################################################################################Why is it reset every episode???? it shud keep the old Q-vals!!
            #if episode > NUM_EPISODE - 9:
            #print("Agent takes action ", action)
            # Take the action on env, so that it returns rewards and new states!
            new_state, reward, done = env.step(action)

            # Update Q-table for Q(s, a)
            q_table[state][action] = q_table[state][action] * (1 - learning_rate) + learning_rate * (
                        reward + discount_rate * np.max(q_table[new_state]))

            # Set new state (transition to next state)
            rewards_current_episode += reward
            state = new_state
            if done:
                acts = {0: 'left', 1: 'right', 2: 'up', 3: 'down'}
                print("*****Episode ", episode)
                print("Agent made to timestep ", timestep)
                print("Actions = ", actions)
                print("Foods = ", food_positions)
                snake_body_coordinates = [(0, SPACE_SIZE), (0, 0)]
                # snake will terminate the game if snake head touches the first row, y?############################################################################
                # go to QLearningEnv.py -> step(self, action), u forgot to pop the tail after each timestep?? so thats y ur snake ends the game when it touches first row??
                direction = "down"
                timestep_index = 0
                snake_scores = 0
                img = np.zeros((env.window_width, env.window_height, 3), dtype=np.uint8)
                head_x = snake_body_coordinates[0][0]
                head_y = snake_body_coordinates[0][1]
                cv2.rectangle(img, (0, 0), (0 + SPACE_SIZE, 0 + SPACE_SIZE), (0, 128, 0),
                              -1)  # SNAKE_HEAD = yellow, snake_body = green, food = white
                cv2.rectangle(img, (head_x, head_y), (head_x + SPACE_SIZE, head_y + SPACE_SIZE), (0, 255, 255), -1)
                cv2.circle(img, (food_positions[timestep_index][0] + SPACE_SIZE // 2,
                                 food_positions[timestep_index][1] + SPACE_SIZE // 2), SPACE_SIZE // 2, (255, 255, 255),
                           -1)
                print("Scores = ", snake_scores)
                while True:
                    cv2.imshow("Snake AI Gameplay", img)

                    if timestep_index > timestep:
                        print("Game over...")
                        cv2.destroyAllWindows()
                        break

                    print("Agent taking action ", acts[actions[timestep_index]])

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

                    if snake_body_coordinates[0][0] == food_positions[timestep_index - 1][0] and \
                            snake_body_coordinates[0][1] == food_positions[timestep_index - 1][1]:
                        snake_scores += 1
                        print("Scores = ", snake_scores)
                        ############################################################################ After eating the food, new food won't be spawned (GO testCV2_SNAKE, it is working. but here isn't working, maybe becoz of the `actions` and `food_pos` arent storing the same things that we are expecting)
                        ##################################################### try to represent states by including 2 more states:
                        # vertical distance of head to food : 0 to 24
                        # horizontal distance of head to food : 0 to 24
                        # 2^12 * 25 * 25 = 2560000
                        print("food_positions[timestep_index][0]", food_positions[timestep_index][0])
                        print("food_positions[timestep_index+1][0]", food_positions[timestep_index + 1][0])
                        cv2.circle(img, (food_positions[timestep_index + 1][0] + SPACE_SIZE // 2,
                                         food_positions[timestep_index + 1][1] + SPACE_SIZE // 2), SPACE_SIZE // 2,
                                   (255, 255, 255), -1)  # update the food
                        # if the food is eaten, the tail won't be popped
                        head = True
                        for body_x, body_y in snake_body_coordinates:
                            if head:
                                cv2.rectangle(img, (body_x, body_y), (body_x + SPACE_SIZE, body_y + SPACE_SIZE),
                                              (0, 255, 255), -1)
                                head = False
                            else:
                                cv2.rectangle(img, (body_x, body_y), (body_x + SPACE_SIZE, body_y + SPACE_SIZE),
                                              (0, 128, 0), -1)

                    else:
                        # move the tail by painting previous position as black
                        cv2.rectangle(img, (snake_body_coordinates[-1][0], snake_body_coordinates[-1][1]), (
                        snake_body_coordinates[-1][0] + SPACE_SIZE, snake_body_coordinates[-1][1] + SPACE_SIZE),
                                      (0, 0, 0), -1)
                        snake_body_coordinates.pop()
                        head = True
                        for body_x, body_y in snake_body_coordinates:
                            if head:
                                cv2.rectangle(img, (body_x, body_y), (body_x + SPACE_SIZE, body_y + SPACE_SIZE),
                                              (0, 255, 255), -1)
                                head = False
                            else:
                                cv2.rectangle(img, (body_x, body_y), (body_x + SPACE_SIZE, body_y + SPACE_SIZE),
                                              (0, 128, 0), -1)

                    if cv2.waitKey(2000) & 0xFF == ord('q'):
                        break
                cv2.destroyAllWindows()
                break

            """if timestep == NUM_TIMESTEP:
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
    
                    # actions[i]  # take all the action, updates the cv2 window, modify the `img`
                    0: 'left',
                    1: 'right',
                    2: 'up',
                    3: 'down'
    
                    # print("Taking action ", actions_names[actions[timestep_index]])
                    # print()
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
    
                    if snake_body_coordinates[0][0] == food_positions[timestep_index-1][0] and snake_body_coordinates[0][1] == food_positions[timestep_index-1][1]:
                        snake_scores += 1
                        print("Scores = ", snake_scores)
                        ############################################################################ After eating the food, new food won't be spawned (GO testCV2_SNAKE, it is working. but here isn't working, maybe becoz of the `actions` and `food_pos` arent storing the same things that we are expecting)
                        ##################################################### try to represent states by including 2 more states:
                        # vertical distance of head to food : 0 to 24
                        # horizontal distance of head to food : 0 to 24
                        # 2^12 * 25 * 25 = 2560000
                        print("food_positions[timestep_index][0]", food_positions[timestep_index][0])
                        print("food_positions[timestep_index+1][0]", food_positions[timestep_index+1][0])
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
    
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break
                cv2.destroyAllWindows()"""

                # visualize the game

        epsilon *= EPS_DECAY
        # Add new reward
        episode_rewards.append(rewards_current_episode)
        episode_points.append(QLearningEnv.score)
        actions.clear()
        food_positions.clear()

    print(episode_rewards[NUM_EPISODE-30:])
    print(episode_points[NUM_EPISODE-30:])
    episode_num = np.array([i for i in range(1, NUM_EPISODE+1)])
    episode_points = np.array(episode_points)
    print("Highest point = ", np.max(episode_points))
    episode_rewards = np.array(episode_rewards)
    print("Highest reward = ", np.max(episode_rewards))
    f1 = plt.figure()
    plt.plot(episode_num, episode_rewards, "bo", label="Rewards")
    f2 = plt.figure()
    plt.plot(episode_num, episode_points, "go", label="Points")
    plt.show()


# go https://www.youtube.com/watch?v=je0DdS0oIZk&ab_channel=TechTribe
# to see how the states of snake can be represented