import cv2
import numpy as np
from queue import PriorityQueue

# circle
x_c = 300
y_c = 185
dia = 80
rad = dia/2

# arrow
x1, y1 = 36, 185
x2, y2 = 115, 210
x3, y3 = 80, 180
x4, y4 = 105, 100
m_ul = (y2 - y1)/(x2 - x1)
m_ur = (y3 - y2) / (x3 - x2)
m_dl = (y1 - y4) / (x1 - x4)
m_dr = (y4 - y3) / (x4 - x3)
m_mid = (y3 - y1) / (x3 - x1)  

# hexagon
w = 70
rad = w/np.sqrt(3)
x5, y5 = 200, 100 + rad
x6, y6 = 200 + (w/2), 100 + rad/2
x7, y7 = 200 + (w/2), 100 - rad/2
x8, y8 = 200, 100 - rad
x0, y0 = 200 - (w/2), 100 + rad/2
x9, y9 = 200 - (w/2), 100 - rad/2
hex_m = (y5 - y0) / (x5 - x0)

# create obstacles with clearance
def circle(x, y):
    circ_eq = ((x - x_c) ** 2 + (y - y_c) ** 2 - rad ** 2) < 0
    return circ_eq

def circle_boundry(x, y):
    circ_eq1 = ((x - x_c) ** 2 + (y - y_c) ** 2 - rad ** 2) >= 0
    circ_eq2 = ((x - x_c) ** 2 + (y - y_c) ** 2 - (rad + 5) ** 2) < 0
    return circ_eq1  and circ_eq2

def hexa_boundry(x, y):
    edg_1 = (-hex_m * x - (-hex_m * x8) + y8 - 5 - y) <= 0 #dl
    edg_2 = (hex_m * x - (hex_m * x8) + y8 - 5 - y) <= 0 #dr
    edg_3 = (-hex_m * x - (-hex_m * x5) + y5 + 5  - y) >= 0 #ur 
    edg_4 = (hex_m * x - (hex_m * x5) + y5 + 5 - y) >= 0 #ul

    edg_5 = (x6 + 5 - x) >= 0 # left
    edg_6 = (x0 - 5 - x) <= 0 # right

    return edg_1 and edg_2 and edg_3 and edg_4 and edg_5 and edg_6

def hexa(x, y):
    edg_1 = (-hex_m * x - (-hex_m * x8) + y8 - y) <= 0 #dl
    edg_2 = (hex_m * x - (hex_m * x8) + y8 - y) <= 0 #dr
    edg_3 = (-hex_m * x - (-hex_m * x5) + y5 - y) >= 0 #ur 
    edg_4 = (hex_m * x - (hex_m * x5) + y5 - y) >= 0 #ul

    edg_5 = (x6 - x) >= 0 # left
    edg_6 = (x0 - x) <= 0 # right

    return edg_1 and edg_2 and edg_3 and edg_4 and edg_5 and edg_6

def arrow_boundry(x,y):
    side_1 = (m_ul * x - m_ul*x2 + y2 + 5 - y) >= 0
    side_2 = (m_ur * x - m_ur*x2 + y2 - 5 - y) <= 0
    side_3 = (m_dl * x - m_dl*x4 + y4 - 7 - y) <= 0
    side_4 = (m_dr * x - m_dr*x4 + y4 + 15 - y) >= 0
    min_line = (m_mid * x - m_mid * x3 + y3 - y) <= 0
    return (side_1 and side_2 and min_line) or (side_3 and side_4 and not min_line)

def arrow(x, y):
    side_1 = (m_ul * x - m_ul*x2 + y2 - y) >= 0
    side_2 = (m_ur * x - m_ur*x2 + y2 - y) <= 0
    side_3 = (m_dl * x - m_dl*x4 + y4 - y) <= 0
    side_4 = (m_dr * x - m_dr*x4 + y4 - y) >= 0
    min_line = (m_mid * x - m_mid * x3 + y3 - y) <= 0
    return (side_1 and side_2 and min_line) or (side_3 and side_4 and not min_line)

def backtrack(parent, goal, start):
    final_path = []
    node = goal
    while node != start:
        node = parent[node]
        final_path.append(node)
    return final_path

# Action sets= {(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1),(1,-1),(-1,-1)}
def get_children(parent_node, image):
    child_list = []
    # right up
    idx = (parent_node[0]+1, parent_node[1]+1)
    if idx[0] < image.shape[0] and idx[1] < image.shape[1] and (image[idx[0], idx[1]][2] == 255):
        child_list.append((idx, 1.4))

    # left up
    idx = (parent_node[0]+1, parent_node[1]-1)
    if idx[0] < image.shape[0] and idx[1] >= 0 and (image[idx[0], idx[1]][2] == 255):
        child_list.append((idx, 1.4))

    # right down
    idx = (parent_node[0]-1, parent_node[1]+1)
    if idx[0] >= 0 and idx[1] < image.shape[1] and (image[idx[0], idx[1]][2] == 255):
        child_list.append((idx, 1.4))

    # left down
    idx = (parent_node[0]-1, parent_node[1]-1)
    if idx[0] >= 0 and idx[1] >= 0 and (image[idx[0], idx[1]][2] == 255):
        child_list.append((idx, 1.4))

    # up
    idx = (parent_node[0]+1, parent_node[1])
    if idx[0] < image.shape[0] and (image[idx[0], idx[1]][2] == 255):
        child_list.append((idx, 1))

    # down
    idx = (parent_node[0]-1, parent_node[1])
    if idx[0] >= 0 and (image[idx[0], idx[1]][2] == 255):
        child_list.append((idx, 1))

    # right
    idx = (parent_node[0], parent_node[1]+1)
    if idx[1] < image.shape[1] and (image[idx[0], idx[1]][2] == 255):
        child_list.append((idx, 1))

    # left
    idx = (parent_node[0], parent_node[1]-1)
    if idx[1] >= 0 and (image[idx[0], idx[1]][2] == 255):
        child_list.append((idx, 1))

    return child_list

def djk (image, start, goal, c2c):
    visited = [start]
    parent_nodes = {}
    c2c[start[0], start[1]] = 0
    que = PriorityQueue()
    que.put((0, start))
    
    if (start== goal):
        parent_nodes[start] = goal
        visited.append(start)
        return True, parent_nodes, visited, c2c

    while (que):
        # get the curr_node with the lowest cost and which hasn't been visited yet
        cur_cost, curr_node = que.get()
        if curr_node == goal:
            return True, parent_nodes, visited, c2c

        children = get_children(curr_node, image)
        for child, child_cost in children:
            if child not in visited:
                # add cost with the child cost with the parent cost
                cur_child_cost =  child_cost + cur_cost
                visited.append(child)
                if c2c[child[0], child[1]] > cur_child_cost:
                    c2c[child[0], child[1]] = cur_child_cost
                    que.put((cur_child_cost, child))
                    parent_nodes[child] = curr_node

    return False, parent_nodes, visited, c2c

# initialize grid 
w = np.ones([400, 250, 3], dtype = np.uint8)
w = 255 * w

# create a cost to come matrix to store costs
c2c_matrix = np.ones([400, 250, 1], dtype = np.uint8)
c2c_matrix = float('inf')*c2c_matrix

# add obstacles on the grid and update cost of obstacle
for x_pos in range(w.shape[0]):
    for y_pos in range(w.shape[1]):
        # check if point is in any of the obstacle space
        if(hexa_boundry(x_pos, y_pos) or (circle_boundry(x_pos, y_pos)) or
             arrow_boundry(x_pos, y_pos) or x_pos < 5 or y_pos < 5 or x_pos > w.shape[0]-5 or y_pos > w.shape[1]-5):
            w[x_pos, y_pos] = [66, 176, 245]
            c2c_matrix[x_pos, y_pos] = -1
        if (circle(x_pos, y_pos) or hexa(x_pos, y_pos) or arrow(x_pos, y_pos)):
            w[x_pos, y_pos] = [0 , 0, 160]
            c2c_matrix[x_pos, y_pos] = -1

# initialize start and goal 
def getStart():
    x = int(input('Enter x coordinate for start point: '))
    y = int(input('Enter y coordinate for start point: '))
    return (x, y)

def getGoal():
    x = int(input('Enter x coordinate for goal point: '))
    y = int(input('Enter y coordinate for goal point: '))
    return (x, y)
print("Find the shortest path in a", w.shape[0],"x", w.shape[1], "space with a clearance of 5mm.")

start = getStart()
goal = getGoal()

# start = (5, 5)
# goal = (10, 10)

#check if goal not is not in obstacle space
if w[start[0], start[1]][2] < 255 or start[0] > w.shape[0] or start[0] < 0  or start[1] > w.shape[1] or start[1] < 0:
    print("The start point ", start, "is not a valid point. Please enter a new start point")

elif w[goal[0], goal[1]][2] < 255 or goal[0] > w.shape[0] or goal[0] < 0  or goal[1] > w.shape[1] or goal[1] < 0:
    print("The goal point ", goal, "is not a valid point. Please enter a new goal point")

else:
    w[start[0], start[1]] = [255, 0, 255]
    w[goal[0], goal[1]] = [255, 0, 255]

    print("Valid start and goal points. Searching for shortest path...")
    # start searching for shortest path
    flag, parents, visited, c2c = djk (w, start, goal, c2c_matrix)
    if (flag):
        print("Path Found. Backtracking...")
        shortest_path = backtrack(parents, goal, start)
        print("Done, shortest path found in", len(shortest_path), "steps.")
        print("Starting Visualizaiton!")

        # animate nodes explored
        for vis in visited:
            w[int(vis[0]), int(vis[1]), :] = [255, 0, 0]
            img = w.copy()
            # img = cv2.resize(img, (1000, 1600), interpolation = cv2.INTER_AREA)
            img = cv2.flip(img, 0)
            img = cv2.flip(img, 1)
            img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
            cv2.imshow("Grid", img)
            cv2.waitKey(1)

        # show path
        for node in shortest_path:
            w[int(node[0]), int(node[1]), :] = [0, 255, 0]

            # Show the final path on grid    
            img = w.copy()
            # img = cv2.resize(img, (1000, 1600), interpolation = cv2.INTER_AREA)
            img = cv2.flip(img, 0)
            img = cv2.flip(img, 1)
            img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
            cv2.imshow("Grid", img)
            cv2.waitKey(1)
        # cv2.imwrite("results/test_case_1/Exploration.png", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No path found!")
