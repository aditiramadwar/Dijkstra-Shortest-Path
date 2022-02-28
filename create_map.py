import cv2
import numpy as np
map_x = 400
map_y = 250
circle_diameter = 80
circle_radius = int(circle_diameter / 2)
circle_pos_center_x = 185
circle_pos_center_y = 300

def circle_obstacle(x, y, radius, map_circle):
    for j in range(x - radius, x + radius):
        for i in range(y - radius, y + radius):
            if float(j - x) ** 2 + float(i - y) ** 2 <= radius ** 2:
                map_circle[i][j] = 0
    return map_circle

map = np.ones((map_y, map_x))
map = circle_obstacle(circle_pos_center_y, circle_pos_center_x, circle_radius, map)

map = 255 * map
map = map.astype(np.uint8)
map = cv2.cvtColor(map, cv2.COLOR_GRAY2RGB)

map = cv2.flip(map, 0)
cv2.imshow("Map", map)

cv2.waitKey(0)
cv2.destroyAllWindows()
