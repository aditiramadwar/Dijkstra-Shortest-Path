import numpy as np
import heapq as hq
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
# create obstacles with clearance
def circle(x, y):
    circ_eq = ((x - 300) ** 2 + (y - 185) ** 2 - 40 * 40) < 0
    return circ_eq

def backtrack(image, parent, goal, start):
    final_path = []
    node = goal
    while node != start:
        node = parent[node]
        final_path.append(node)
        image[node[0], node[1]] = [0, 255, 0]
    return final_path, image

def get_children(parent_node, image):
    child_list = []
    # right up
    idx = (parent_node[0]+1, parent_node[1]+1)
    if idx[0] < image.shape[0] and idx[1] < image.shape[1]:
            child_list.append((idx, 1.4))

    # left up
    idx = (parent_node[0]+1, parent_node[1]-1)
    if idx[0] < image.shape[0] and idx[1] >= 0:
            child_list.append((idx, 1.4))

    # right down
    idx = (parent_node[0]-1, parent_node[1]+1)
    if idx[0] >= 0 and idx[1] < image.shape[1]:
            child_list.append((idx, 1.4))

    # left down
    idx = (parent_node[0]-1, parent_node[1]-1)
    if idx[0] >= 0 and idx[1] >= 0:
            child_list.append((idx, 1.4))

    # up
    idx = (parent_node[0]+1, parent_node[1])
    if idx[0] < image.shape[0]:
            child_list.append((idx, 1))

    # down
    idx = (parent_node[0]-1, parent_node[1])
    if idx[0] >= 0:
            child_list.append((idx, 1))

    # right
    idx = (parent_node[0], parent_node[1]+1)
    if idx[1] < image.shape[1]:
            child_list.append((idx, 1))

    # left
    idx = (parent_node[0], parent_node[1]-1)
    if idx[1] >= 0:
            child_list.append((idx, 1))

    return child_list

def djk (image, start, goal):
    que = []
    visited = []
    nodes_cost = {}
    parent_nodes = {}

    nodes_cost[start] = 0
    hq.heappush(que, (0, start))
    while (len(que) > 0):
        # get the curr_node with the lowest cost and which hasn't been visited yet
        cur_cost, curr_node = hq.heappop(que)
        if curr_node not in visited:
            visited.append(curr_node)
            children = get_children(curr_node, image)
            for child, child_cost in children:
                image[child[0], child[1]] = [255, 0, 0]
                if child not in visited:
                    # add cost with the child cost with the parent cost
                    cur_child_cost =  child_cost + cur_cost

                    # if this cost is less than the prev stored cost of this nodex  
                    # then update the cost and the parent node
                    if child in nodes_cost:
                        if cur_child_cost < nodes_cost[child]:
                            nodes_cost[child] = cur_child_cost
                            parent_nodes[child] = curr_node
                            # update the queue with the new smaller cost of the node
                            hq.heapreplace(que, (cur_child_cost, child))
                            hq.heapify(que)
                    # if it is a new node being explored, add it to the dictionary
                    else:
                        nodes_cost[child] = cur_child_cost
                        parent_nodes[child] = curr_node
                        # append the node in queue
                        hq.heappush(que, (cur_child_cost, child))
                        hq.heapify(que)

                    if child == goal:
                        return True, image, nodes_cost, parent_nodes
    return False, image, nodes_cost, parent_nodes

# initialize grid 
w = np.ones([400, 250, 3], dtype = np.uint8)
w = 255 * w

# add obstacles on the grid
for x_pos in range(w.shape[0]):
    for y_pos in range(w.shape[1]):
        if (circle(x_pos, y_pos)):
            w[x_pos, y_pos] = [0, 0, 160]

# initialize start and goal 
start = (0, 0)
goal = (10, 10)

# start searching for shortest path
flag, w, cost, parents = djk (w, start, goal)
if (flag):
    print("Path Found")
    path, w = backtrack(w, parents, goal, start)
    print("Shortest Path: ", path)

w = ndimage.rotate(w, 90)
imgplot = plt.imshow(w)
plt.show()