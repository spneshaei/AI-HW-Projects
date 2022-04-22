# Estimating functions using Genetic methods
# Seyed Parsa Neshaei - 98106134

import numpy as np
import matplotlib.pyplot as pplot
import time
from random import *

# Functions

def check_for_khatkhati_value(x):
    inputs = [ 0.1, 0.11, 0.12,  0.13,  0.14,  0.15,  0.16,  0.17,  0.18,  0.19,  0.2,  0.21,
  0.22,  0.23,  0.24,  0.25,  0.26,  0.27,  0.28,  0.29,  0.3,   0.31,  0.32,  0.33,
  0.34,  0.35,  0.36,  0.37,  0.38,  0.39,  0.4,   0.41,  0.42,  0.43,  0.44,  0.45,
  0.46,  0.47,  0.48,  0.49,  0.5,   0.51,  0.52,  0.53,  0.54,  0.55,  0.56,  0.57,
  0.58,  0.59,  0.6,   0.61,  0.62,  0.63,  0.64,  0.65,  0.66,  0.67,  0.68,  0.69,
  0.7,   0.71,  0.72,  0.73,  0.74,  0.75,  0.76,  0.77,  0.78,  0.79,  0.8,   0.81,
  0.82,  0.83,  0.84,  0.85,  0.86,  0.87,  0.88,  0.89]
    outputs = [  7.67371504e-03,   4.37169934e-04,   2.38796185e-03,   5.43527404e-03,
   2.53085029e-04,   8.65587439e-03,   1.45490630e-02,   1.09811645e-02,
   6.29101030e-03,   1.24610415e-02,   1.14069459e-02,  1.82928440e-02,
   1.07101901e-02,   4.24106583e-03,   5.02378509e-03, 2.58554146e-02,
   4.33256522e-03,   4.33320325e-03,   2.62303576e-02,   4.74593290e-03,
   3.59497257e-02,   3.95461134e-02,   3.24827040e-02,   8.92475928e-02,
   3.59819123e-02,   4.98243511e-02,   3.37922778e-03,   1.80645589e-02,
   5.04874588e-03,   4.53474677e-02,   3.65218254e-02,   5.27260038e-02,
   5.94004648e-03,   6.52918498e-02,   2.89278853e-02,   7.33100918e-02,
   6.97017174e-02,   2.27328989e-02,   2.45215207e-02,   6.85281724e-02,
   2.75513179e-02,   9.71100979e-02,   5.55168259e-02,   2.21682439e-02,
   5.12670420e-02,   1.99928107e-02,   1.44276555e-01,   6.71492389e-02,
   2.05762237e-02,   9.60205711e-02,   7.70483949e-02,   5.15186129e-02,
   1.71952177e-01,   3.38825950e-02,   2.44665741e-02,   1.20442788e-01,
   1.99624375e-01,   2.31938725e-02,   1.16371939e-01,   7.44134903e-02,
   1.30717781e-01,   6.60966440e-02,   1.36364461e-01,   1.13188898e-01,
   1.47396235e-01,   2.61828086e-02,   1.02391481e-01,   8.06815794e-02,
   1.32130254e-01,   1.48753625e-01,   1.00394590e-01,   1.17316204e-01,
   1.50769410e-01,   1.38108854e-01,   1.20107826e-01,   3.86935559e-01,
   3.29525748e-01,   3.73480007e-01,   2.44826152e-01,   2.91320360e-02]
    for i in range(len(inputs)):
        if abs(inputs[i] - x) < 1e-9:
            return outputs[i]
    return 0

def check_for_not_continuous_value(x):
    if x < 0.5:
        return x + 0.7
    if x >= 0.5 and x < 0.7:
        return x * x * x + 0.7
    if x >= 0.7:
        return x * x + 1

def khatkhati_function(x):
    return map(check_for_khatkhati_value, x)

def not_continuous_function(x):
    return map(check_for_not_continuous_value, x)

# No-Error versions of normal np functions

def my_log(parameter):
    if parameter < 1:
        return 0
    try:
        return np.log(parameter)
    except:
        return 0

def my_exp(parameter):
    try:
        return np.where(parameter > 8, np.exp(8), np.exp(parameter))
    except:
        return 0

def my_sqrt(parameter):
    try:
        return np.sqrt(np.abs(parameter))
    except:
        return 0

def my_add_with_one(parameter):
    return np.add(parameter, 1)

def my_divide(parameter1, parameter2):
    if parameter2 == 0:
        return 0
    try:
        return np.divide(parameter1, parameter2)
    except:
        return 0

def my_power(parameter1, parameter2):
    try:
        return np.power(parameter1, parameter2)
    except:
        return 0

def my_inv(parameter):
    try:
        return np.where(parameter == 0, 0, 1 / parameter)
    except:
        return 0

# Classified list of functions

UNARY_FUNCTIONS = {'addwithone': my_add_with_one, 'log': my_log, 'sin': np.sin, 'cos': np.cos, 'exp': my_exp, 'sqrt': my_sqrt, 'inv': my_inv}
BINARY_FUNCTIONS = {'+': np.add, '*': np.multiply, '-': np.subtract,  '/': my_divide, '^': my_power} 

# Genetic Parameters
population_size = 1500
max_count_of_iterations = 1500
initial_tree_depth = 3
max_tree_depth = 15
percent_to_choose_each_time = 30
number_to_choose_each_iteration = percent_to_choose_each_time *  population_size // 100
fitness_evaluation_method = 'abs' # abs | square

# Counter Variables
fitness_calculation_count = 0

# Tree definition and methods
class Tree:
    def __init__(self, depth, isFull = True):
        self.depth = depth
        self.isFull = isFull # is our tree full (balanced depth up to last row, each node with two children except the leaves) or not?
        self.accuracy = 0.0 # 1 / fitness
        self.nodes = []
        self.populateNodes()

    def calculateValueOnVariableValue(self, variable_value, from_node_index):
        global all_trees
        global UNARY_FUNCTIONS
        global BINARY_FUNCTIONS
        global population_size
        global max_count_of_iterations
        global initial_tree_depth
        global max_tree_depth
        global percent_to_choose_each_time
        global number_to_choose_each_iteration
        global fitness_evaluation_method

        # Idea to return from_node_index also to better handle children of a node's index values is from one of the references linked in the report file sent in CW alongside this
        from_node = self.nodes[from_node_index]
        if from_node != 'x':
            if from_node in list(UNARY_FUNCTIONS.keys()):
                child_value, new_from_node_index = self.calculateValueOnVariableValue(variable_value, from_node_index + 1)
                return UNARY_FUNCTIONS.get(from_node)(child_value), new_from_node_index
            elif from_node in list(BINARY_FUNCTIONS.keys()):
                left_child_value, new_from_node_index = self.calculateValueOnVariableValue(variable_value, from_node_index + 1)
                right_child_value, new_from_node_index = self.calculateValueOnVariableValue(variable_value, new_from_node_index + 1)
                return BINARY_FUNCTIONS.get(from_node)(left_child_value, right_child_value), new_from_node_index
        else:
            return variable_value, from_node_index

    def findExpression(self, from_node_index):
        global all_trees
        global UNARY_FUNCTIONS
        global BINARY_FUNCTIONS
        global population_size
        global max_count_of_iterations
        global initial_tree_depth
        global max_tree_depth
        global percent_to_choose_each_time
        global number_to_choose_each_iteration
        global fitness_evaluation_method

        from_node = self.nodes[from_node_index]
        if from_node != 'x':
            if from_node in list(UNARY_FUNCTIONS.keys()):
                child_value, new_from_node_index = self.findExpression(from_node_index + 1)
                return ' ' + from_node + '(' + child_value + ') ', new_from_node_index
            elif from_node in list(BINARY_FUNCTIONS.keys()):
                left_child_value, new_from_node_index = self.findExpression(from_node_index + 1)
                right_child_value, new_from_node_index = self.findExpression(new_from_node_index + 1)
                return '(' + left_child_value + ' ' + from_node + ' ' + right_child_value  + ')', new_from_node_index
        else:
            return 'x', from_node_index

    def calculateDepth(self, from_node_index):
        global all_trees
        global UNARY_FUNCTIONS
        global BINARY_FUNCTIONS
        global population_size
        global max_count_of_iterations
        global initial_tree_depth
        global max_tree_depth
        global percent_to_choose_each_time
        global number_to_choose_each_iteration
        global fitness_evaluation_method
        from_node = self.nodes[from_node_index]
        if from_node != 'x':
            if from_node in list(UNARY_FUNCTIONS.keys()):
                child_depth, new_from_node_index = self.calculateDepth(from_node_index + 1)
                return child_depth + 1, new_from_node_index # +1 due to current node
            elif from_node in list(BINARY_FUNCTIONS.keys()):
                left_child_depth, new_from_node_index = self.calculateDepth(from_node_index + 1)
                right_child_depth, new_from_node_index = self.calculateDepth(new_from_node_index)
                final_child_depth = left_child_depth if left_child_depth > right_child_depth else right_child_depth
                return final_child_depth + 1, new_from_node_index # +1 due to current node
        else:
            # only node here is x
            return 1, from_node_index + 1

    def calculateSquareAccuracy(self, input_values, output_values):
        global all_trees
        global UNARY_FUNCTIONS
        global BINARY_FUNCTIONS
        global population_size
        global max_count_of_iterations
        global initial_tree_depth
        global max_tree_depth
        global percent_to_choose_each_time
        global number_to_choose_each_iteration
        global fitness_evaluation_method
        square_difference = 0
        numeber_of_inputs = len(input_values)
        for i in range(numeber_of_inputs):
            input_value = input_values[i]
            output_value = output_values[i]
            tree_value, _ = self.calculateValue(input_value)
            square_difference += (output_value - tree_value) ** 2
        return square_difference / numeber_of_inputs

    def calculateAbsAccuracy(self, input_values, output_values):
        global all_trees
        global UNARY_FUNCTIONS
        global BINARY_FUNCTIONS
        global population_size
        global max_count_of_iterations
        global initial_tree_depth
        global max_tree_depth
        global percent_to_choose_each_time
        global number_to_choose_each_iteration
        global fitness_evaluation_method
        abs_difference = 0
        numeber_of_inputs = len(input_values)
        for i in range(numeber_of_inputs):
            input_value = input_values[i]
            output_value = output_values[i]
            tree_value, _ = self.calculateValueOnVariableValue(input_value, 0)
            abs_difference += abs(output_value - tree_value)
        return abs_difference / numeber_of_inputs

    def calculateAccurary(self, input_values, output_values): # Calculate fitness
        global fitness_calculation_count
        global all_trees
        global UNARY_FUNCTIONS
        global BINARY_FUNCTIONS
        global population_size
        global max_count_of_iterations
        global initial_tree_depth
        global max_tree_depth
        global percent_to_choose_each_time
        global number_to_choose_each_iteration
        global fitness_evaluation_method

        fitness_calculation_count += 1
        if len(input_values) > 0:
            if fitness_evaluation_method == 'square':
                self.accuracy = self.calculateSquareAccuracy(input_values, output_values)
            elif fitness_evaluation_method == 'abs':
                self.accuracy = self.calculateAbsAccuracy(input_values, output_values)
            return self.accuracy
        else:
            return 999999999 # return some big num because if it is small then the tree would survive!

    def populateNodes(self):
        global all_trees
        global UNARY_FUNCTIONS
        global BINARY_FUNCTIONS
        global population_size
        global max_count_of_iterations
        global initial_tree_depth
        global max_tree_depth
        global percent_to_choose_each_time
        global number_to_choose_each_iteration
        global fitness_evaluation_method
        if self.isFull:
            self.populateNodesFully()
        else:
            self.populateNodesPartially()

    def addRandomBinaryNode(self):
        global all_trees
        global UNARY_FUNCTIONS
        global BINARY_FUNCTIONS
        global population_size
        global max_count_of_iterations
        global initial_tree_depth
        global max_tree_depth
        global percent_to_choose_each_time
        global number_to_choose_each_iteration
        global fitness_evaluation_method
        self.nodes.append(choice(list(BINARY_FUNCTIONS.keys())))

    def addRandomUnaryNode(self):
        global all_trees
        global UNARY_FUNCTIONS
        global BINARY_FUNCTIONS
        global population_size
        global max_count_of_iterations
        global initial_tree_depth
        global max_tree_depth
        global percent_to_choose_each_time
        global number_to_choose_each_iteration
        global fitness_evaluation_method
        self.nodes.append(choice(list(UNARY_FUNCTIONS.keys())))

    def addVariable(self):
        global all_trees
        global UNARY_FUNCTIONS
        global BINARY_FUNCTIONS
        global population_size
        global max_count_of_iterations
        global initial_tree_depth
        global max_tree_depth
        global percent_to_choose_each_time
        global number_to_choose_each_iteration
        global fitness_evaluation_method
        self.nodes.append('x')

    def populateNodesPartially(self, current_depth = 0):
        global all_trees
        global UNARY_FUNCTIONS
        global BINARY_FUNCTIONS
        global population_size
        global max_count_of_iterations
        global initial_tree_depth
        global max_tree_depth
        global percent_to_choose_each_time
        global number_to_choose_each_iteration
        global fitness_evaluation_method
        if self.depth != current_depth:
            probability_to_add_variable = random()
            if probability_to_add_variable > 0.5:
                self.addVariable()
            else:
                should_insert_binary = random()
                if should_insert_binary < 0.5:
                    self.addRandomBinaryNode()
                    self.populateNodesFully(current_depth + 1)
                    self.populateNodesFully(current_depth + 1)
                else:
                    self.addRandomUnaryNode()
                    self.populateNodesFully(current_depth + 1)
        else:
             self.addVariable()

    def populateNodesFully(self, current_depth = 0):
        global all_trees
        global UNARY_FUNCTIONS
        global BINARY_FUNCTIONS
        global population_size
        global max_count_of_iterations
        global initial_tree_depth
        global max_tree_depth
        global percent_to_choose_each_time
        global number_to_choose_each_iteration
        global fitness_evaluation_method
        if self.depth != current_depth:
            should_insert_binary = random()
            if should_insert_binary < 0.5:
                self.addRandomBinaryNode()
                self.populateNodesFully(current_depth + 1)
                self.populateNodesFully(current_depth + 1)
            else:
                self.addRandomUnaryNode()
                self.populateNodesFully(current_depth + 1)
        else:
            self.addVariable()

all_trees = []

def initializeAllTrees():
    global all_trees
    global UNARY_FUNCTIONS
    global BINARY_FUNCTIONS
    global population_size
    global max_count_of_iterations
    global initial_tree_depth
    global max_tree_depth
    global percent_to_choose_each_time
    global number_to_choose_each_iteration
    global fitness_evaluation_method

    all_trees = []
    for count in range(population_size):
        all_trees.append(Tree(initial_tree_depth, random() > 0.7)) # in 30% of times, isFull is True

def findAllAccuracies(input_values, output_values):
    global all_trees
    global UNARY_FUNCTIONS
    global BINARY_FUNCTIONS
    global population_size
    global max_count_of_iterations
    global initial_tree_depth
    global max_tree_depth
    global percent_to_choose_each_time
    global number_to_choose_each_iteration
    global fitness_evaluation_method
    count = len(all_trees)
    for i in range(count):
        all_trees[i].calculateAccurary(input_values, output_values)

def bestInSample(sample): # Idea to choose best ones from one of the resources noted in the report
    global all_trees
    global UNARY_FUNCTIONS
    global BINARY_FUNCTIONS
    global population_size
    global max_count_of_iterations
    global initial_tree_depth
    global max_tree_depth
    global percent_to_choose_each_time
    global number_to_choose_each_iteration
    global fitness_evaluation_method
    sample_count = len(sample)
    best_tree = sample[0]
    for i in range(1, sample_count):
        if best_tree.accuracy > all_trees[i].accuracy:
            best_tree = all_trees[i]
    return best_tree

def findTree():
    global all_trees
    global UNARY_FUNCTIONS
    global BINARY_FUNCTIONS
    global population_size
    global max_count_of_iterations
    global initial_tree_depth
    global max_tree_depth
    global percent_to_choose_each_time
    global number_to_choose_each_iteration
    global fitness_evaluation_method
    number_to_choose_each_iteration_best_ones = sample(all_trees, number_to_choose_each_iteration) # random.sample
    return bestInSample(number_to_choose_each_iteration_best_ones)

def findVariableInTree(tree, node_index):
    global all_trees
    global UNARY_FUNCTIONS
    global BINARY_FUNCTIONS
    global population_size
    global max_count_of_iterations
    global initial_tree_depth
    global max_tree_depth
    global percent_to_choose_each_time
    global number_to_choose_each_iteration
    global fitness_evaluation_method
    if tree.nodes[node_index] in list(UNARY_FUNCTIONS.keys()):
        return findVariableInTree(tree, node_index + 1)
    if tree.nodes[node_index] in list(BINARY_FUNCTIONS.keys()):
        return findVariableInTree(tree, findVariableInTree(tree, node_index + 1))
    if tree.nodes[node_index] == 'x':
        return 1 + node_index

def performCrossover(first, second):
    global all_trees
    global UNARY_FUNCTIONS
    global BINARY_FUNCTIONS
    global population_size
    global max_count_of_iterations
    global initial_tree_depth
    global max_tree_depth
    global percent_to_choose_each_time
    global number_to_choose_each_iteration
    global fitness_evaluation_method
    randomNodeFromFirst = randint(0, len(first.nodes) - 1)
    variableNodeFromFirst = findVariableInTree(first, randomNodeFromFirst)
    randomNodeFromSecond = randint(0, len(second.nodes) - 1)
    variableNodeFromSecond = findVariableInTree(second, randomNodeFromSecond)
    newChild = Tree(first.depth, True)
    newChild.nodes = []
    newChild.nodes += second.nodes[0:randomNodeFromSecond]
    newChild.nodes += first.nodes[randomNodeFromFirst:variableNodeFromFirst]
    newChild.nodes += second.nodes[variableNodeFromSecond:len(second.nodes)]
    if newChild.depth < max_tree_depth:
        return newChild
    return newChild if random() < (1/4) else Tree(first.depth, True)

def performMutation(tree):
    global all_trees
    global UNARY_FUNCTIONS
    global BINARY_FUNCTIONS
    global population_size
    global max_count_of_iterations
    global initial_tree_depth
    global max_tree_depth
    global percent_to_choose_each_time
    global number_to_choose_each_iteration
    global fitness_evaluation_method
    if random() > 0.9:
        return tree
    node_to_be_replaced_index = randint(0, len(tree.nodes) - 1)
    if tree.nodes[node_to_be_replaced_index] in list(UNARY_FUNCTIONS.keys()):
        tree.nodes[node_to_be_replaced_index] = choice(list(UNARY_FUNCTIONS.keys()))
    elif tree.nodes[node_to_be_replaced_index] in list(BINARY_FUNCTIONS.keys()):
        tree.nodes[node_to_be_replaced_index] = choice(list(BINARY_FUNCTIONS.keys()))
    return tree

def addNewTreeToPopulation(new_tree):
    global all_trees
    global UNARY_FUNCTIONS
    global BINARY_FUNCTIONS
    global population_size
    global max_count_of_iterations
    global initial_tree_depth
    global max_tree_depth
    global percent_to_choose_each_time
    global number_to_choose_each_iteration
    global fitness_evaluation_method
    worst_fitness_tree = all_trees[0]
    all_trees_count = len(all_trees)
    for i in range(all_trees_count):
        worst_fitness_tree = worst_fitness_tree if worst_fitness_tree.accuracy > all_trees[i].accuracy else all_trees[i]
    if worst_fitness_tree.accuracy < all_trees[0].accuracy:
        return
    for i in range(all_trees_count):
        if all_trees[i].accuracy == worst_fitness_tree.accuracy:
            all_trees[i] = new_tree
            break

def estimate(input_values, output_values):
    global fitness_calculation_count
    global all_trees
    global UNARY_FUNCTIONS
    global BINARY_FUNCTIONS
    global population_size
    global max_count_of_iterations
    global initial_tree_depth
    global max_tree_depth
    global percent_to_choose_each_time
    global number_to_choose_each_iteration
    global fitness_evaluation_method
    findAllAccuracies(input_values, output_values)
    hasPrintedCount = False
    for count in range(max_count_of_iterations):
        first = findTree()
        second = findTree()
        new_tree = performMutation(performCrossover(first, second))
        new_tree.calculateAccurary(input_values, output_values)
        addNewTreeToPopulation(new_tree)
        if new_tree.accuracy < 1e-9:
            print("Count of generations: " + str(count))
            print("Count of fitness calculations: " + str(fitness_calculation_count))
            hasPrintedCount = True
            break
    if not hasPrintedCount:
        print("Count of generations: " + str(max_count_of_iterations))
        print("Count of fitness calculations: " + str(fitness_calculation_count))
    answer = all_trees[0]
    all_trees_count = len(all_trees)
    for i in range(all_trees_count):
        answer = answer if answer.accuracy < all_trees[i].accuracy else all_trees[i]
    return answer

np.seterr(all='raise')

to_be_estimated_functions = [(lambda x: x**2), (lambda x: x**3), (lambda x: x**4), (lambda x: x**2 + x), (lambda x: np.sin(x) + x), (lambda x: np.log(x) + np.cos(x) - x), (lambda x: np.sqrt(x) * np.sin(x)), (lambda x: np.sqrt(x) / np.cos(x)), (lambda x: 2 / x), (lambda x: np.tan(x) + 1), (lambda x: x + np.pi), (lambda x: map(check_for_khatkhati_value, x)), (lambda x: map(check_for_not_continuous_value, x))]

to_be_estimated_function = to_be_estimated_functions[0]
input_values = np.arange(0.1, 0.9, 0.01)
y = to_be_estimated_function(input_values)

start_time = time.time()
initializeAllTrees()
answer = estimate(input_values, y)
end_time = time.time()

print("Expression: y(x) = " + answer.findExpression(0)[0])
print("Fitness: " + str('infinity' if answer.accuracy == 0 else 1 / answer.accuracy))
print("Time: " + str(end_time - start_time) + " seconds")
y_pred = [[answer.calculateValueOnVariableValue(x, 0)[0]] for x in input_values]
pplot.plot(input_values, y, color='g')
pplot.plot(input_values, y_pred, color='b')
pplot.show()