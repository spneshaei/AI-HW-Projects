from __future__ import print_function # TODO: Remove if it makes error - included due to Python version problems: https://github.com/sylhare/nprime/issues/4
from random import randrange, seed
import xml.etree.ElementTree as ET

node_details = []

class Node:
    i = 0
    j = 0
    f = 0
    g = 0
    h = 0
    parent = None

    def __init__(self, i, j, parent = None):
        self.i = i
        self.j = j
        self.f = 0
        self.g = 0
        self.h = 0
        self.parent = parent

    def is_equal_to(self, node):
        return (self.i == node.i) and (self.j == node.j)

    def is_in_list(self, given_list):
        for node in given_list:
            if (node.i == self.i) and (node.j == self.j):
                return True
        return False

    def populate_f(self):
        self.f = self.g + self.h

    def neighbors(self):
        neighbors_list = []
        if self.i > 0 and node_details[self.i - 1][self.j] != "obstacle":
            neighbors_list.append(Node(self.i - 1, self.j, self))
        if self.i < len(node_details) - 1 and node_details[self.i + 1][self.j] != "obstacle":
            neighbors_list.append(Node(self.i + 1, self.j, self))
        if self.j > 0 and node_details[self.i][self.j - 1] != "obstacle":
            neighbors_list.append(Node(self.i, self.j - 1, self))
        if self.j < len(node_details[0]) - 1 and node_details[self.i][self.j + 1] != "obstacle":
            neighbors_list.append(Node(self.i, self.j + 1, self))
        return neighbors_list

def find_min_f_in_array(array):
    min_node = array[0]
    for node in array:
        if node.f < min_node.f:
            min_node = node
    return min_node

def print_grid(current, battery):
    print("\n")
    for i in range(0, len(node_details)):
        print('|', end="")
        for j in range(0, len(node_details[i])):
            if current.is_equal_to(Node(i, j)):
                print("*", end="|")
                continue
            if battery.is_equal_to(Node(i, j)):
                print("B", end="|")
                continue
            if node_details[i][j] == "obstacle":
                print("#", end="|")
                continue
            if node_details[i][j] == "start":
                print("S", end="|")
                continue
            if node_details[i][j] == "goal":
                print("B", end="|")
                continue
            print(' ', end="|")
        print("\n", end="")

def a_star_search(robot, battery, heuristic, printing = True, should_find_distance = True):
    number_of_checks = 0
    distance = 0
    explored = []
    robot.g = 0
    robot.h = heuristic(robot, battery)
    robot.populate_f()
    frontier = [robot]
    current_node = Node(robot.i, robot.j)
    while len(frontier) > 0:
        number_of_checks += 1
        new_current_node = find_min_f_in_array(frontier)
        if should_find_distance:
            distance += real_heuristic(current_node, new_current_node) - 1
        current_node = new_current_node
        explored.append(current_node)
        frontier.remove(current_node)
        if printing:
            print_grid(current_node, battery)
        if current_node.is_equal_to(battery):
            return distance, number_of_checks, get_path_to(current_node)
        for node in current_node.neighbors():
            if not node.is_in_list(explored): # Dont revisit explored, avoid looping forever
                if node.is_in_list(frontier): # then we may have found a new, less 'g' so it should be updated
                    if node.g > current_node.g + 1:
                        node.g = current_node.g + 1
                        node.populate_f()
                        node.parent = current_node
                else:
                    node.g = current_node.g + 1
                    node.h = heuristic(node, battery)
                    node.populate_f()
                    node.parent = current_node
                    frontier.append(node)
    return distance, number_of_checks, []

def manhattan_heuristic(from_node, to_node):
    return abs(from_node.i - to_node.i) + abs(from_node.j - to_node.j)

def real_heuristic(from_node, to_node):
    _, _, path = a_star_search(Node(from_node.i, from_node.j), Node(to_node.i, to_node.j), manhattan_heuristic, printing = False, should_find_distance = False)
    return len(path)

def vertical_distance_heuristic(from_node, to_node):
    return abs(from_node.i - to_node.i)

def horizontal_distance_heuristic(from_node, to_node):
    return abs(from_node.j - to_node.j)

def random_heuristic(from_node, to_node):
    return randrange(0, 30)

def no_heuristics(from_node, to_node):
    return 0

def dfs_search(robot, battery):
    distance = 0
    number_of_checks = 0
    stack = []
    stack.append(robot)
    explored = []
    current_node = Node(robot.i, robot.j)
    while len(stack) > 0:
        number_of_checks += 1
        new_current_node = stack.pop()
        distance += real_heuristic(current_node, new_current_node) - 1
        current_node = new_current_node
        explored.append(current_node)
        print_grid(current_node, battery)
        if current_node.is_equal_to(battery):
            return distance, number_of_checks, get_path_to(current_node)
        for node in current_node.neighbors():
            if not node.is_in_list(explored):
                stack.append(node)
    return distance, number_of_checks, []

def get_path_to(node):
    path = []
    while node.parent is not None:
        path.append(node)
        node = node.parent
    path.append(node) # start node (with no parent)
    return path[::-1]

def is_node_in_path(node, path):
    for n in path:
        if n.is_equal_to(node):
            return True
    return False

def print_path(path, alg_name, robot, battery, number_of_checks, distance):
    print(alg_name + ' Path\n--------------\nLength of path: ' + str(len(path)) + '\nNumber of checks: ' + str(number_of_checks)+ '\nDistance navigated: ' + str(distance))
    print("\n")
    for i in range(0, len(node_details)):
        print('|', end="")
        for j in range(0, len(node_details[i])):
            if battery.is_equal_to(Node(i, j)):
                print("B", end="|")
                continue
            if robot.is_equal_to(Node(i, j)):
                print("S", end="|")
                continue
            if is_node_in_path(Node(i, j), path):
                print("*", end="|")
                continue
            if node_details[i][j] == "obstacle":
                print("#", end="|")
                continue
            print(' ', end="|")
        print("\n", end="")
    
    for node in path:
        print("( i = " + str(node.i) + ", j = " + str(node.j) + " )"
            + (' -> robot' if node.is_equal_to(robot) else (' -> battery' if node.is_equal_to(battery) else '')))
    print("\n")

seed(20)
xml_files = [
    'SampleRoom.xml',
    'SampleRoom2.xml',
    'SampleRoom3.xml', 
    'SampleRoom4.xml',
    'SampleRoom5.xml',
    'SampleRoom6.xml',
]
for xml_file in xml_files:
    node_details = []
    print('File: ' + xml_file + '\n------------------------------------------')
    xml_rows = ET.parse(xml_file).getroot()
    robot = Node(0, 0)  # initial temp vals
    battery = Node(0, 0) # initial temp vals
    i = 0
    for xml_row in xml_rows:
        row = []
        j = 0
        for xml_cell in xml_row:
            if xml_cell.text == 'robot':
                robot.i = i
                robot.j = j
                row.append("empty")
            elif xml_cell.text == 'Battery':
                battery.i = i
                battery.j = j
                row.append("empty")
            else:
                row.append(xml_cell.text)
            j += 1
        i += 1
        node_details.append(row)

    print("A* with Manhattan Heuristic Movements\n--------------")
    d, number_of_checks_astart, path_astar = a_star_search(robot, battery, manhattan_heuristic)
    print_path(path_astar, 'A* with Manhattan Heuristic search', robot, battery, number_of_checks_astart, d)

    print("A* with Real Heuristic Movements\n--------------")
    d, number_of_checks_astart_real, path_astar_real = a_star_search(robot, battery, manhattan_heuristic)
    print_path(path_astar_real, 'A* with Real Heuristic search', robot, battery, number_of_checks_astart_real, d)

    print("A* with Vertical Distance Heuristic Movements\n--------------")
    d, number_of_checks_astart_ver, path_astar_ver = a_star_search(robot, battery, vertical_distance_heuristic)
    print_path(path_astar_ver, 'A* with Vertical Distance Heuristic search', robot, battery, number_of_checks_astart_ver, d)

    print("A* with Horizontal Distance Heuristic Movements\n--------------")
    d, number_of_checks_astart_hor, path_astar_hor = a_star_search(robot, battery, horizontal_distance_heuristic)
    print_path(path_astar_hor, 'A* with Horizontal Distance Heuristic search', robot, battery, number_of_checks_astart_hor, d)

    print("A* with Random Heuristic Movements\n--------------")
    d, number_of_checks_astart_ran, path_astar_ran = a_star_search(robot, battery, random_heuristic)
    print_path(path_astar_ran, 'A* with Random Heuristic search', robot, battery, number_of_checks_astart_ran, d)

    print("A* with No Heuristic Movements\n--------------")
    d, number_of_checks_astart_no_h, path_astar_no_h = a_star_search(robot, battery, no_heuristics)
    print_path(path_astar_no_h, 'A* with No Heuristic search', robot, battery, number_of_checks_astart_no_h, d)

    print("DFS Movements\n--------------")
    d, number_of_checks_dfs, path_dfs = dfs_search(robot, battery)
    print_path(path_dfs, 'DFS Search', robot, battery, number_of_checks_dfs, d)


