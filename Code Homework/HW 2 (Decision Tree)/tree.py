import numpy as np
import pandas as ps
from sklearn.model_selection import train_test_split
import gvgen
from graphviz import Source

def findNumberOfMatchings(node):
    node.number_of_matches = np.sum(node.data.T[-1])
    node.total_number = node.data.shape[0]

def fillNodeDetails(node):
    try:
        node.entropy = findEntropy(node.number_of_matches, node.total_number - node.number_of_matches)
    except RuntimeWarning:
        node.entropy = 0
    node.title = "Yes" if node.number_of_matches > node.total_number - node.number_of_matches else "No"

def addRowsToMap(rows, map, index):
    for row in rows:
        if row[index] in map:
            map[row[index]].append(row)
        else:
            map[row[index]] = [row]

def trainChildren(node):
    if len(node.children) == 0:
        return
    for key, value in node.children.items():
        trainFromNode(value)

def setChildren(other_features_map, node):
    node.removeChildren()
    for key, value in other_features_map.items(): # add new one as a node
        node.children[key] = Node(np.array(value))
        node.children[key].checked_features = node.checked_features + [node.value]
        node.children[key].depth = node.depth + 1

def findEntropy(number_of_matches, number_of_mismatches):
    total_number = number_of_matches + number_of_mismatches
    q_param = number_of_matches / total_number
    return -1 * (np.log2(q_param) * q_param + np.log2(1 - q_param) * (1 - q_param)) if number_of_matches != 0 and number_of_matches != total_number else 0

def trainFromNode(node):
    other_features_map = {}
    findNumberOfMatchings(node)
    fillNodeDetails(node)
    if node.number_of_matches == 0 or node.number_of_matches == node.total_number or node.depth == 8: # assumed maxDepth = 8
        return
    for index in range(len(node.data[0]) - 1):
        if not index in node.checked_features:
            addRowsToMap(node.data, other_features_map, index)
            node.calculateStatsForFeature(other_features_map.items(), index)
            if index == node.value:
                setChildren(other_features_map, node)
            other_features_map = {}
    trainChildren(node)
    if len(node.children) == 0: # leaf, should be yes/no
        node.title = "Yes" if node.number_of_matches > (node.total_number - node.number_of_matches) else "No"

def representation(node):
    if len(node.children) == 0:
        return node.title
    return node.title + "\n" + "\nEntropy " + str(node.entropy) + "\nGain " + str(node.gain)

def categorize(data):
    global categories
    for i in range(len(features) - 1):
        categories.append(ps.cut(data[i], 6, retbins=True)[1])
        data[i] = np.digitize(data[i], categories[i], right=True)

def gvGen(gv, node):
    newNode = gv.newItem(representation(node))
    if node.isChild():
        return newNode
    else:
        for key, value in node.children.items():
            if isRestaurant:
                gv.propertyAppend(gv.newLink(newNode, gvGen(gv, value)), "label", key)
            else:
                current_categorization = categories[value.value]
                gv.propertyAppend(gv.newLink(newNode, gvGen(gv, value)), "label", "[" + str(round(current_categorization[int(key) - 1], 2)) + ", " + str(round(current_categorization[int(key)], 2)) + "]")
        return newNode

def saveGraph(node):
    gv = gvgen.GvGen()
    gvGen(gv, node)
    gv.dot
    temp1 = open("temp.txt", 'w')
    gv.dot(temp1)
    temp1.close()
    temp2 = open("temp.txt", 'r')
    lines = temp2.readlines()[1:]
    src = Source("".join(lines))
    src.render()

def findAccuracy(data, node):
    global number_of_corrects
    number_of_corrects = 0
    total = len(data)
    for entry in data:
        findAccuracyOfEntry(entry, node)
    return 100 * number_of_corrects / total

def findAccuracyOfEntry(entry, root):
    global number_of_corrects
    if len(root.children) == 0:
        findAccuracyOfLeaf(entry, root)
    else:
        if entry[root.value] in root.children.keys():
            findAccuracyOfEntry(entry, root.children[entry[root.value]])
        else:
            number_of_corrects += int(np.random.rand() < 0.5)

def findAccuracyOfLeaf(entry, root):
    global number_of_corrects
    real_result = entry[-1]
    if (root.number_of_matches > root.total_number - root.number_of_matches and real_result == 1) or (root.number_of_matches < root.total_number - root.number_of_matches and real_result == 0):
        number_of_corrects += 1

class Node:
    def __init__(self, training_data):
        self.data = training_data
        self.children = {}
        self.checked_features = []
        self.depth = 0
        self.value = -1
        self.title = ""
        self.description_on_graph = ""
        self.entropy = -1
        self.gain = -1
        self.number_of_matches = 0
        self.total_number = 0

    def isChild(self):
        return len(self.children) == 0

    def removeChildren(self):
        self.children.clear()

    def setGain(self, remainder, value):
        global features
        if self.entropy - remainder > self.gain:
            self.gain = self.entropy
            self.title = features[value]
            self.value = value
            self.gain -= remainder

    def calculateStatsForFeature(self, difference, attribute):
        remainder = 0
        for _, values in difference:
            number_of_matches, number_of_mismatches = 0, 0
            for value in values:
                number_of_matches += 1 if value[-1] == 1 else 0
                number_of_mismatches += 1 if value[-1] != 1 else 0
            total_number = number_of_matches + number_of_mismatches
            remainder += (total_number / self.total_number) * findEntropy(number_of_matches, number_of_mismatches)
        self.setGain(remainder, attribute)

isRestaurant = True # or False, if diabetes is target
features = []
categories = [] # (bins)
number_of_corrects = 0

csv = ps.read_csv('restaurant.csv' if isRestaurant else 'diabetes.csv')
features, data = csv.columns, csv.to_numpy().T
if not isRestaurant:
    categorize(data)

if isRestaurant:
    training = data.T
else:
    training, testing = train_test_split(data.T, test_size = 20 / 100)

node = Node(np.array(training))
trainFromNode(node)

trainingAccuracy = findAccuracy(training, node)
print("Training accuracy " + str(trainingAccuracy) + "%")

if not isRestaurant:
    testAccuracy = findAccuracy(testing, node)
    print("Test accuracy " + str(testAccuracy) + "%")

saveGraph(node)
