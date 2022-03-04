import numpy as np
import heapq as hq
import cv2
import copy
# create obstacles with clearance
def circle(x, y):
    circ_eq = ((x - 300) ** 2 + (y - 185) ** 2 - 40 * 40) < 0
    return circ_eq

def circle_boundry(x, y):
    circ_eq1 = ((x - 300) ** 2 + (y - 185) ** 2 - 40 * 40) >= 0
    circ_eq2 = ((x - 300) ** 2 + (y - 185) ** 2 - 45 * 45) < 5
    return circ_eq1  and circ_eq2

def hexa_boundry(x, y):
    edg_1 = (-0.571 * x + 169.286 - y) <= 0 # down left
    edg_2 = (-0.571 * x + 259.286 - y) >= 0 # up right

    edg_3 = (0.571 * x + 30.714 - y) >= 0
    edg_4 = (0.571 * x - 59.286 - y) <= 0
    
    edg_5 = (240 - x) >= 0 # point 7 # right 
    edg_6 = (160 - x) <= 0 # point 9 # left
    return edg_1 and edg_2 and edg_3 and edg_4 and edg_5 and edg_6
def quad_boundry(x,y):

    side_1 = (0.316 * x + 173.608 - y) >= -5 # up left
    side_2 = (0.857 * x + 111.429 - y) <= 7 #up right
    min_line = (-0.114 * x + 189.091 - y) <= 0
    side_3 = (-3.2 * x + 436 - y) >= -15
     # down right
    side_4 = (-1.232 * x + 229.348 - y) <= 10 #down left
    return (side_1 and side_2 and min_line) or (side_3 and side_4 and not min_line)

def hexa(x, y):
    edg_1 = (-0.571 * x + 174.286 - y) <= 0
    edg_2 = (-0.571 * x + 254.286 - y) >= 0

    edg_3 = (0.571 * x + 25.714 - y) >= 0
    edg_4 = (0.571 * x - 54.286 - y) <= 0
    
    edg_5 = (235 - x) >= 0 # point 7
    edg_6 = (165 - x) <= 0 # point 9
    return edg_1 and edg_2 and edg_3 and edg_4 and edg_5 and edg_6

def quad(x, y):
    side_1 = (0.316 * x + 173.608 - y) >= 0
    side_2 = (0.857 * x + 111.429 - y) <= 0
    min_line = (-0.114 * x + 189.091 - y) <= 0
    side_3 = (-3.2 * x + 436 - y) >= 0
    side_4 = (-1.232 * x + 229.348 - y) <= 0



    return (side_1 and side_2 and min_line) or (side_3 and side_4 and not min_line)

def backtrack(image, parent, goal, start):
    final_path = []
    node = goal
    while node != start:
        node = parent[node]
        final_path.append(node)
    return final_path, image

def get_children(parent_node, image):
    child_list = []
    # right up
    idx = (parent_node[0]+1, parent_node[1]+1)
    if idx[0] < image.shape[0] and idx[1] < image.shape[1]:
        if (image[idx[0], idx[1]][2] == 255):
            child_list.append((idx, 1.4))

    # left up
    idx = (parent_node[0]+1, parent_node[1]-1)
    if idx[0] < image.shape[0] and idx[1] >= 0:
        if (image[idx[0], idx[1]][2] == 255):
            child_list.append((idx, 1.4))

    # right down
    idx = (parent_node[0]-1, parent_node[1]+1)
    if idx[0] >= 0 and idx[1] < image.shape[1]:
        if image[idx[0], idx[1]][2] == 255:
            child_list.append((idx, 1.4))

    # left down
    idx = (parent_node[0]-1, parent_node[1]-1)
    if idx[0] >= 0 and idx[1] >= 0:
        if image[idx[0], idx[1]][2] == 255:
            child_list.append((idx, 1.4))

    # up
    idx = (parent_node[0]+1, parent_node[1])
    if idx[0] < image.shape[0]:
        if image[idx[0], idx[1]][2] == 255:
            child_list.append((idx, 1))

    # down
    idx = (parent_node[0]-1, parent_node[1])
    if idx[0] >= 0:
        if image[idx[0], idx[1]][2] == 255:
            child_list.append((idx, 1))

    # right
    idx = (parent_node[0], parent_node[1]+1)
    if idx[1] < image.shape[1]:
        if image[idx[0], idx[1]][2] == 255:
            child_list.append((idx, 1))

    # left
    idx = (parent_node[0], parent_node[1]-1)
    if idx[1] >= 0:
        if image[idx[0], idx[1]][2] == 255:
            child_list.append((idx, 1))

    return child_list

def djk (image, start, goal):
    que = []
    visited = [start]
    nodes_cost = {}
    parent_nodes = {}
    nodes_cost[start] = 0
    hq.heappush(que, (0, start))
    while (len(que) > 0):
        # get the curr_node with the lowest cost and which hasn't been visited yet
        cur_cost, curr_node = hq.heappop(que)
        if curr_node == goal:
                return True, image, nodes_cost, parent_nodes, visited

        children = get_children(curr_node, image)
        for child, child_cost in children:
                if child not in visited:
                    # add cost with the child cost with the parent cost
                    cur_child_cost =  child_cost + cur_cost
                    visited.append(child)

                    # if this cost is less than the prev stored cost of this nodex  
                    # then update the cost and the parent node
                    if child in nodes_cost:
                        if cur_child_cost < nodes_cost[child]:
                            nodes_cost[child] = cur_child_cost
                            # update the queue with the new smaller cost of the node
                            hq.heapreplace(que, (cur_child_cost, child))
                            hq.heapify(que)
                    # if it is a new node being explored, add it to the dictionary
                    else:
                        nodes_cost[child] = cur_child_cost
                        # append the node in queue
                        hq.heappush(que, (cur_child_cost, child))
                        hq.heapify(que)

                    parent_nodes[child] = curr_node

    return False, image, nodes_cost, parent_nodes, visited

# initialize grid 
w = np.ones([400, 250, 3], dtype = np.uint8)
w = 255 * w

# add obstacles on the grid
for x_pos in range(w.shape[0]):
    for y_pos in range(w.shape[1]):
        # check if point is in any of the obstacle space
        if(hexa_boundry(x_pos, y_pos) or (circle_boundry(x_pos, y_pos)) or quad_boundry(x_pos, y_pos)):
            w[x_pos, y_pos] = [66, 176, 245]
        if (circle(x_pos, y_pos) or hexa(x_pos, y_pos) or quad(x_pos, y_pos)):
            w[x_pos, y_pos] = [0 , 0, 160]


rgb_w = cv2.resize(w, (1000, 1600), interpolation = cv2.INTER_AREA)
rgb_w = cv2.flip(rgb_w, 0)
rgb_w = cv2.flip(rgb_w, 1)
rgb_w = cv2.rotate(rgb_w, cv2.cv2.ROTATE_90_CLOCKWISE)
cv2.imshow("Explored region", rgb_w)
cv2.waitKey(0)
cv2.destroyAllWindows()
# initialize start and goal 
start = (0, 0)
goal = (399, 249)

#check if goal not is not in obstacle space
if w[goal[0], goal[1]][2] < 255:
    print("IN OBSTACLE!!")
else:
        print("Valid goal. Searching for shortest path...")
        # start searching for shortest path
        flag, img, cost, parents, visited = djk (w, start, goal)
        if (flag):
                print("Path Found. Backtracking...")
                shortest_path, img = backtrack(img, parents, goal, start)
                print("Done")
                # animate nodes explored
                for vis in visited:
                        w[int(vis[0]), int(vis[1]), :] = [255, 0, 0]
                        img = w.copy()

                        img = cv2.resize(img, (1000, 1600), interpolation = cv2.INTER_AREA)
                        img = cv2.flip(img, 0)
                        img = cv2.flip(img, 1)
                        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
                        cv2.imshow("Grid", img)
                        cv2.waitKey(1)

                # show path
                for node in shortest_path:
                        w[int(node[0]), int(node[1]), :] = [0, 255, 0]
                img = w.copy()
                img = cv2.resize(img, (1000, 1600), interpolation = cv2.INTER_AREA)
                img = cv2.flip(img, 0)
                img = cv2.flip(img, 1)
                img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
                cv2.imshow("Grid", img)
                cv2.waitKey(0)
        else:
            print("No path found!")
cv2.destroyAllWindows()



    