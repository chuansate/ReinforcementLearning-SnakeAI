import cv2
import numpy as np
x = 0
y = 0
SIZE = 20
# Create a black image
img = np.zeros((500,500,3), np.uint8)
while True:
    cv2.rectangle(img, (x,y), (x + SIZE, y + SIZE), (255, 0, 0), -1)
    cv2.imshow("Snake", img)
    y += SIZE
    cv2.rectangle(img, (x, y-SIZE), (x + SIZE, y), (0, 0, 0), -1)
    # img[y-SIZE:y][x:x+SIZE] = (0, 0, 0)
    # cv2.rectangle(img, (x - SIZE, y - SIZE), (x, y), (0, 0, 0), -1)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
