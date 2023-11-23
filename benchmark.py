'''
    Source Code for Algorithms Project for COT6405

<<<<<<< HEAD
    Algorithms for Brute Force, Nearest Neighor, Branch and Bound, Dynamic programming are 
    referenced from GeeksForGeeks to ensure accurate implementation. Datasets are either auto generated or sourced from
    The University of Waterloo in Waterloo, Ontario, Canada
    2-opt is referenced from KeiruaProd to ensure accurate implementation.
=======
    Algorithms for Brute Force, Nearest Neighor, and Dynamic programming are 
    referenced from GeeksForGeeks to ensure accurate implementation. Branch and bround is
    referenced from Abdul Bari's guide on youtube linked below.
    Datasets are from The University of Waterloo.
>>>>>>> origin/master

    References: 
    BruteForce - https://www.geeksforgeeks.org/traveling-salesman-problem-tsp-implementation/
    Branch and Bound - https://www.youtube.com/watch?v=1FEP_sNb62k&t=515s
    Dynamic Programming - https://www.geeksforgeeks.org/travelling-salesman-problem-using-dynamic-programming/
    2-Opt - https://www.keiruaprod.fr/blog/2021/09/15/traveling-salesman-with-2-opt.html
    Christofides - 
        Prim's algorithm portion of Christofides - https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/
        Other aspects of Christofides - 
            https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.operators.binary.compose.html
            http://matejgazda.com/christofides-algorithm-in-python/
            https://www.youtube.com/watch?v=Uu2ptesvteE
            https://notebook.community/DhashS/Olin-Complexity-Final-Project/code/03_approximation_algorithms
            https://networkx.org/documentation/stable/reference/convert.html
    Greedy - https://www.geeksforgeeks.org/travelling-salesman-problem-greedy-approach/
    Ant Colony - 
        1. https://induraj2020.medium.com/implementation-of-ant-colony-optimization-using-python-solve-traveling-salesman-problem-9c14d3114475#:~:text=Implementing%20Ant%20colony%20optimization%20in%20python%2D%20solving%20Traveling%20salesman%20problem,-Induraj&text=Ant%20colony%20optimization%20(ACO)%20is,by%20the%20behavior%20of%20ants.
        2. https://github.com/Josephbakulikira/Traveling-Salesman-Algorithm/blob/master/antColony.py 
        3. https://www.youtube.com/watch?v=EJKdmEbGre8
    Simulated Annealing Sources - 
        1. https://medium.com/swlh/how-to-implement-simulated-annealing-algorithm-in-python-ab196c2f56a0
        2. https://machinelearningmastery.com/simulated-annealing-from-scratch-in-python/
        3. https://github.com/perrygeo/simanneal/blob/master/simanneal/anneal.py 
        4. https://nathanrooy.github.io/posts/2020-05-14/simulated-annealing-with-python/ 
        5. https://towardsdatascience.com/optimization-techniques-simulated-annealing-d6a4785a1de7
        6. https://www.baeldung.com/cs/simulated-annealing 
        7. https://www.youtube.com/watch?v=35fzyblVdmA
    Datasets - http://www.math.uwaterloo.ca/tsp/world/countries.html
'''

# Import required libraries
import sys
import argparse
import math
import time 
import matplotlib.pyplot as plt
import os
import heapq
import networkx
import random
from math import *
from random import *
from numpy import *
from itertools import permutations
from func_timeout import func_timeout, FunctionTimedOut
from copy import deepcopy
from networkx import *
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import *


def plot_path(coords, path, algorithm, total_distance, time):
    
    # Unpack the coordinates list of tuples into two seperate list, x, and y, zip groups first elements (x-coordinates together, and second elements (y-coordinates together)
    x, y = zip(*coords)

    # Create a scatter plot of the cities, takes two arguments, x coordinates, and y-coordinates and plots these points on a 2D graph.
    plt.scatter(x,y)

    # Plot the path taken
    for i in range(len(path) - 1): # Stop at one less than the final city because final city will try to connect to a city out of bounds
        plt.plot((coords[path[i]][0], coords[path[i+1]][0]), (coords[path[i]][1], coords[path[i+1]][1]), 'r-') # First argument is a tuple containing the x coordinates of the first and second city, the second tuple is for the y coordinates respectively
    plt.plot((coords[path[-1]][0], coords[path[0]][0]), (coords[path[-1]][1], coords[path[0]][1]), 'r-') # Connect the final city in the path to the first city in the path

    # Add title
    plt.title(f"{algorithm.replace('_', ' ').title()} - Cities: {len(coords)} - Distance (Euclidean): {round(total_distance, 4)} - Time (Seconds): {time}")

    # Add labels
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")

    # Show the plot
    plt.show()

def generate_distance_matrix(coords, self_distance = 0):
    num_cities = len(coords)
    '''
    Generate a euclidean distance matrix for a list of coordinate tuples.

    The self distance for every city to itself should be infinity for Branch and Bound to avoid an algorithm considering loops which return to the same city.
    However, it is 0 by default for other algorithms.

    The inner loop creates a row equal in length to the number of cities, and the router loop causes this loop
    to execute a number of times equal to the number of cities. This results in a matrix with an equal number of 
    rows and columns.
    '''
    matrix = [[self_distance for city in range(num_cities)] for city in range(num_cities)] 
     # Calculate the distance for each city i to every other city
    for i in range(num_cities):
        for j in range(num_cities): 
            if i != j: # Any city traveling to itself should have a distance of 0, for example city 0 -> city 0 results in cost of 0
                matrix[i][j] = euclidean_distance(coords[i], coords[j]) # Calculate the euclidean distance for each city pair, and use them to populate the distance matrix
    return matrix

#This calculates total distance for the simulated annealing function
def calc_total_distance(pathTotVal, distTotMatrix): 
    
    sumTotDist = 0
    
    for numValTot in range(len(pathTotVal)-1): 
        
        sumTotDist += distTotMatrix[pathTotVal[i]][pathTotVal[i+1]]
        
    sumTotDist += distTotMatrix[pathTotVal[-1]][pathTotVal[0]]
    
    return sumTotDist

def reduce_matrix(distance_matrix):
    """
        Perform row and column reduction on the matrix and calculate the reduction cost.
    """

    num_cities = len(distance_matrix)
    cost = 0 # Initialize cost of reduction 0 

    # Reduce row
    for row in range(num_cities):
        min_value = min(distance_matrix[row]) # Find the minimum value in the row
        cost += min_value # Add minimum value from each row to the cost
        if min_value != 0: # If min the minimum value is 0, the row is already reduced, no need to enter inner loop
            for column in range(num_cities):
                distance_matrix[row][column] -= min_value # Subtract the minimum value from every element in the row

    # Reduce column
    for column in range(num_cities): # Iterate through each column 
        min_value = min(distance_matrix[row][column] for row in range(num_cities)) # Get the minimum value in the column
        cost += min_value # Add minimum value from each column to the cost
        if min_value != 0: # If min value was zero, no need to enter the inner loop
            for row in range(num_cities):
                distance_matrix[row][column] -= min_value # Subtract the min value from each row in the column

    return distance_matrix, cost # Return reduced distance matrix, aditionally return the cost to be used in calculating the lower bound of a node 

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def read_tsp_file_from_disk(filename):
    try:
        with open(filename, 'r') as f:
            return read_tsp_file(f.read())
    except FileNotFoundError:
        print(f"Error: The file '{filename}' does not exist.")
        exit(1)
    except IOError:
        print(f"Error: The file '{filename}' could not be opened.")
        exit(1)

# Function to read TSP dataset from a string
def read_tsp_file(file_content):
    # Split lines and initialize variables
    lines = file_content.strip().split('\n')
    coords = []
    reading_coords = False
    
    # Loop through each line in TSP dataset
    for line in lines:
        if "DIMENSION" in line:
            # Get the dimension specified in the TSP file, for potential future use
            dimension = int(line.split(":")[1])
        if reading_coords and line != 'EOF':
            # Read coordinates and append to list
            parts = line.split()
            coords.append((float(parts[1]), float(parts[2])))
        if "NODE_COORD_SECTION" in line:
            # Start reading coordinates
            reading_coords = True
    
    return coords

'''
    These 2 functions are used to allow 2-opt to function 
'''
def swapTwo (route, i, j): 
    
    #Check if i and j should be swapped and swap them
    if j < i: 
        
        i, j = j, i
        
    return route[:i] + list(reversed(route[i:j+1])) + route[j+1:]

def distTotal (route, coordins):
    
    distAmount = 0
    
    for item in range(len(route)): 
        
        if item > 0: 
            
            distAmount += math.sqrt(sum((coordins[route[item]][d] - coordins[route[item-1]][d])**2 for d in range(len(coordins[0]))))
            
        
    return distAmount

'''
    
    This mstMinKey function allows for one to find the vertex, that is not yet included, 
    with the smallest distance value.
    
'''

def mstMinKey (coords, mstKeyVal, mstMinSetVals):
    
    mstMinVal = sys.maxsize
    mstMinIndexNum = -1
    
    for kObj in range(len(coords)): 
        
        if mstKeyVal[kObj] < mstMinVal and mstMinSetVals[kObj] == False:
            
            mstMinVal = mstKeyVal[kObj]
            mstMinIndexNum = kObj
            
    return mstMinIndexNum
    

# Brute-force algorithm for solving the TSP
def brute_force_tsp(coords): # Coords represents a list of tuples, the 0th element of any tupel represents the x element, and the 1st element of any tuple represents the y element
    min_distance = float('inf')  # Keep track of shortest distance so far, start at infinity so all paths found will be less than initial value
    best_path = None  # Keep track of the coordinate indices that represent the best path found so far
    
    ''' Take the number of coordinates, generate a range of of numbers from that length, each representing a coordinate index, 
    going from 0 to len(coords) - 1, and generate all permutations of that range, representing all paths we can take through the list of coordinates'''
    for current_path in permutations(range(len(coords))): # Calculate the distance of each permutation or path
        current_distance = 0 # Start with a distance of 0 for each path or permutation
        for i in range(len(current_path) - 1): # We need to start stop before the final location in the path because we access the next index in the loop.
            ''' We calculate the euclidean distance between two coordinates by calling the euclidean_distance function, 
            We pass in two coordinate tuples from the coordinates list, accessing them using the path indices of the current location and the next in the path'''
            current_distance += euclidean_distance(coords[current_path[i]], coords[current_path[i+1]]) 
        current_distance += euclidean_distance(coords[current_path[-1]], coords[current_path[0]]) # Now, we connect the final location in the path to the starting position to complete the cycle
        
        # Update min_distance and best_path
        if current_distance < min_distance: # If the current path distance just calculated is less than the stored minimum we've found so far, we have a new minimum and best path
            min_distance = current_distance # Assign the new minimum euclidean distance to minimum distance
            best_path = current_path # Assign the new best path permutation to best path, the indices representing the coordinates locations in the coords list of tuples 
    
    return best_path, min_distance # Return the optimal answer to the TSP problem 

# Greedy algorithm for solving TSP
def nearest_neighbor_tsp(coords):
    num_cities = len(coords) # The number of coords tuples in the coords list gives us the number of cities the problem
    cities_left_to_visit = set(range(num_cities)) # Keep track of the unvisted cities using a set of integers, each representing the index of a point in the coords list from 0 to len of num_cities - 1
    found_path = [] # Initialize an empty list which will contain the path of city indexes for the coords list
    current_city = 0  # Start from the first city
    found_path.append(current_city) # Add first city to the found path so we can keep track of our starting point
    cities_left_to_visit.remove(current_city) # First city has been visited so it should not be visited again until the very end of the algorithm to complete the hamiltonian cycle
    found_distance = 0 # Keep track of the total distance of the found path, we start at 0 since we are still at the starting position
    
    # Visit nearest unvisited city
    while cities_left_to_visit: # Not all cities have been visited yet so we keep constructing the path
        '''
            To find where we need to go from the current city, we need to look through all of the unvisited cities, and find the one the minimum euclidean distance
            min returns to us the smallest item in an iterable, or the smallest of two or more arguments, in this case, our iterable is the unvisited set
            The key argument is a function that is used to compute a key for each item in our iterable, essentially each city is given a distance key, iterable: 1 - key: 20 
            The key is used to guide the min function in how to make the min comparison, the key for each iterable is the euclidean distance between the current city and that unvisited city
            The min function will then find the item with the smallest key (distance) and return it to the variable nearest_city


            The euclidean_distance function takes two coordinate tuples which are accessed using two indexes, 
            both tuples represent a city's x, y coordinates in the TSP problem. The indexes are for the 
            the current city, and the nearest city's tuple location in the coords list. 
            When the function completes, it returns the euclidean distance between those cities.

            Important Note: min will not consider cities that have been removed from the cities_left_to_visit set at the end of each iteration, meaning a city can't be traveled to twice

        '''
        nearest_city = min(cities_left_to_visit, key=lambda city: euclidean_distance(coords[current_city], coords[city])) 
        found_distance += euclidean_distance(coords[current_city], coords[nearest_city]) # Since we are traveling to next city in the path, we have to add the distance to get there to total distance
        current_city = nearest_city # Adjust our current location to the city we just traveled to
        found_path.append(current_city) # Add the index of the current city to our found path so we know the path which provided our found distance at the end
        cities_left_to_visit.remove(current_city) # Our current city has been visited so we should not travel to it again according to the definition of the TSP problem
    
    found_distance += euclidean_distance(coords[current_city], coords[found_path[0]]) # Now that every city has been visited, we must return from the last city back to the start to complete the hamiltonian cycle
    return found_path, found_distance # Now that we have the path and total distance, we must return them so we can visualize the results

# Branch and Bound algorithm for solving the TSP
def branch_and_bound_tsp(coords):
    
    # Generate the distance matrix from the list of coordinate tuples
    distance_matrix = generate_distance_matrix(coords)

    # Define a node class for our state based tree which will store a node's path, it's reduced matrix, and its cost
    class Node:
        def __init__(self, matrix, cost, path, num_cities, unvisited):
            self.matrix = matrix
            self.cost = cost
            self.path = path
            self.num_cities = num_cities
            self.unvisited = unvisited 

        # Define how we compare a node to another node so they can be correctly inserted into a min heap, and we can always dequeue the node with the lowest cost
        def __lt__(self, other):
            return self.cost < other.cost # Return true if the lowerbound for this node is lower than the one being compared to

        @property # Allow is_complete_tour to be accessed without parenthesis by defining it as a property 
        def is_complete_tour(self):
            return len(self.path) == self.num_cities # Check if tour is complete (node is a leaf node)
 
    best_path, upper_bound = nearest_neighbor_tsp(coords) # Get an upperbound and starting path using NN. All node costs must less than upperbound to avoid being pruned

    distance_matrix, starting_cost = reduce_matrix(distance_matrix)
    starting_node = Node(distance_matrix, starting_cost, [0], len(coords), set(range(len(coords))))
    starting_node.unvisited.remove(0) # Remove starting city from unvisited for starting node
    priority_queue = [starting_node] # Our priority queue ensures we always explore the node with the lowest cost first, and at first containins only the root

    while priority_queue:
        current_node = heapq.heappop(priority_queue)
        if (current_node.is_complete_tour):
            if (current_node.cost < upper_bound):
                upper_bound = current_node.cost + euclidean_distance(coords[current_node.path[-1]], coords[current_node.path[0]]) # Only update upper if leaf node is reached and a better solution is found
                best_path = current_node.path + [0] # Add the starting city to complete the tour
                continue

        for unvisited in current_node.unvisited: 
            child_matrix = deepcopy(current_node.matrix) # Start with a copy of parent matrix
            
            # Set values in the row for the beginning city to infinity
            row = current_node.path[-1]
            for column in range(len(coords)):
                child_matrix[row][column] = float('inf')

            # Set values in the column for the destination city to infinity
            column = unvisited
            for row in range(len(coords)):
                child_matrix[row][column] = float('inf')    

            # Set distance the from child city to the beginning city in the path to infinity
            child_matrix[unvisited][0] = float('inf')

            # Reduce child matrix
            child_matrix, reduced_cost = reduce_matrix(child_matrix)

            # Find cost of child node, add distance from previous city to next city, the cost of the previous node, and the cost to reduce the matrix of the child node
            child_cost = current_node.matrix[current_node.path[-1]][unvisited] + current_node.cost + reduced_cost
            
            # If child cost is less than the upper bound, create a new node and enqueue it, otherwise prune it as it's not worth exploring further
            if (child_cost < upper_bound):
                child_path = current_node.path[:] + [unvisited]
                child_unvisited = deepcopy(current_node.unvisited)
                child_unvisited.remove(unvisited)           
                child_node = Node(child_matrix, child_cost, child_path, len(coords), child_unvisited)
                heapq.heappush(priority_queue, child_node)

    return best_path, upper_bound # Return the best path and minimum cost 

def held_karp_tsp(coords):
    n = len(coords)
    distances = [[euclidean_distance(coords[i], coords[j]) for j in range(n)] for i in range(n)]
    memo = {}

    def hk_recursive(S, j):
        if S == (1 << j):
            return distances[0][j]
        if (S, j) in memo:
            return memo[(S, j)]

        costs = [
            hk_recursive(S & ~(1 << j), k) + distances[k][j]
            for k in range(1, n) if S & (1 << k) and k != j
        ]
        
        if costs:
            memo[(S, j)] = min(costs)
        else:
            memo[(S, j)] = float('inf')  # No valid cost, set to infinity
        
        return memo[(S, j)]

    min_cost = min(hk_recursive((1 << n) - 2, j) + distances[j][0] for j in range(1, n))
    
    # Reconstructing the path
    S, j, path = (1 << n) - 2, 0, [0]
    for _ in range(n-1):
        j = min(
            ((k, distances[j][k] + hk_recursive(S & ~(1 << k), k))
             for k in range(1, n) if S & (1 << k)),
            key=lambda x: x[1]
        )[0]
        S &= ~(1 << j)
        path.append(j)
    
    return path, min_cost

#two-opt function
def two_opt_tsp(coords): 
    
    '''
        Use nearest neighbor to find Initial route. Nearest Neighbor Code copied from code written by Aaron above.
    '''
    
    twoOptImprovement = 1;
    
    num_cities = len(coords) # The number of coords tuples in the coords list gives us the number of cities the problem
    cities_left_to_visit = set(range(num_cities)) # Keep track of the unvisted cities using a set of integers, each representing the index of a point in the coords list from 0 to len of num_cities - 1
    found_path = [] # Initialize an empty list which will contain the path of city indexes for the coords list
    current_city = 0  # Start from the first city
    found_path.append(current_city) # Add first city to the found path so we can keep track of our starting point
    cities_left_to_visit.remove(current_city) # First city has been visited so it should not be visited again until the very end of the algorithm to complete the hamiltonian cycle
    found_distance = 0 # Keep track of the total distance of the found path, we start at 0 since we are still at the starting position
    
    # Visit nearest unvisited city
    while cities_left_to_visit: # Not all cities have been visited yet so we keep constructing the path
        '''
            To find where we need to go from the current city, we need to look through all of the unvisited cities, and find the one the minimum euclidean distance
            min returns to us the smallest item in an iterable, or the smallest of two or more arguments, in this case, our iterable is the unvisited set
            The key argument is a function that is used to compute a key for each item in our iterable, essentially each city is given a distance key, iterable: 1 - key: 20 
            The key is used to guide the min function in how to make the min comparison, the key for each iterable is the euclidean distance between the current city and that unvisited city
            The min function will then find the item with the smallest key (distance) and return it to the variable nearest_city


            The euclidean_distance function takes two coordinate tuples which are accessed using two indexes, 
            both tuples represent a city's x, y coordinates in the TSP problem. The indexes are for the 
            the current city, and the nearest city's tuple location in the coords list. 
            When the function completes, it returns the euclidean distance between those cities.

            Important Note: min will not consider cities that have been removed from the cities_left_to_visit set at the end of each iteration, meaning a city can't be traveled to twice

        '''
        nearest_city = min(cities_left_to_visit, key=lambda city: euclidean_distance(coords[current_city], coords[city])) 
        found_distance += euclidean_distance(coords[current_city], coords[nearest_city]) # Since we are traveling to next city in the path, we have to add the distance to get there to total distance
        current_city = nearest_city # Adjust our current location to the city we just traveled to
        found_path.append(current_city) # Add the index of the current city to our found path so we know the path which provided our found distance at the end
        cities_left_to_visit.remove(current_city) # Our current city has been visited so we should not travel to it again according to the definition of the TSP problem
    
    found_distance += euclidean_distance(coords[current_city], coords[found_path[0]])
    
    '''
        
        Two Opt Begins here. The initial route is the found route from nearest neighbor.
        
    '''
    
    currentRoute = found_path
    currentDist = found_distance
    
    #Keep Looping while there is improvement
    while twoOptImprovement > 0: 
        
        #Set the distance to beat as the current distance
        BeatDist = currentDist
        
        for oneSwap in range(1, len(currentRoute)-2): 
            
            for endSwap in range(oneSwap+1, len(currentRoute)): 
                
                routeNew = swapTwo(currentRoute, oneSwap, endSwap)
                distanceNew = distTotal (routeNew, coords)
                
                if distanceNew < BeatDist: 
                    
                    currentRoute = routeNew
                    currentDist = distanceNew
                    
        twoOptImprovement = 1 - (currentDist/BeatDist)
    
    return currentRoute, currentDist

def christofides_tsp(coords): 
    '''
    This function implements the Christofides algorithm for the Traveling Salesman Problem (TSP).
    It takes as input a list of coordinates and returns a tour and its total weight.
    '''

    # Generate the distance matrix for the given coordinates
    christofidesMatrix = generate_distance_matrix(coords)
    
    # Initialize a new NetworkX graph
    nxGraph = Graph()
    nxGraph.add_nodes_from(range(len(coords)))

    # Add all the edges to the graph, with their weights set to the corresponding distances in christofidesMatrix
    for nxItemOne in range(len(coords)): 
        for nxItemTwo in range(nxItemOne + 1, len(coords)):    
            nxGraph.add_edge(nxItemOne, nxItemTwo, weight=christofidesMatrix[nxItemOne][nxItemTwo])

    # Compute the Minimum Spanning Tree (MST) of the graph
    mst = networkx.minimum_spanning_tree(nxGraph)

    # Add weights to the edges of the MST
    for edge in mst.edges():
        node1, node2 = edge
        mst[node1][node2]['weight'] = christofidesMatrix[node1][node2]

    # Find the vertices with odd degree in the MST
    oddDegrees = []
    for oddVert in mst.nodes(): 
        if mst.degree(oddVert) % 2 != 0: 
            oddDegrees.append(oddVert)

    # Create a graph with the odd degree vertices and add edges between them
    oddDegreeGraph = Graph()
    oddDegreeGraph.add_nodes_from(oddDegrees)
    for i in range(len(oddDegrees)):
        for j in range(i + 1, len(oddDegrees)):
            oddDegreeGraph.add_edge(oddDegrees[i], oddDegrees[j], weight=christofidesMatrix[oddDegrees[i]][oddDegrees[j]])

    # Find the minimum weight perfect matching in the graph of odd degree vertices
    minWeightGraph = max_weight_matching(oddDegreeGraph, maxcardinality = True)

    # Create a new graph for the minimum weight perfect matching
    matchingGraph = Graph()
    for edge in minWeightGraph:
        node1, node2 = edge
        matchingGraph.add_edge(node1, node2, weight=christofidesMatrix[node1][node2])

    # Combine the MST and the minimum weight perfect matching to create an Eulerian graph
    multiGraphGraph = compose(mst, matchingGraph)

    # Check if the graph is connected
    if not is_connected(multiGraphGraph):
        raise NetworkXError("Graph is not connected.")

    # Check if the graph is Eulerian
    if not is_eulerian(multiGraphGraph):
        raise NetworkXError("Graph is not Eulerian.")

    # Find an Eulerian circuit in the graph
    chEulerianCircuit = list(eulerian_circuit(multiGraphGraph))

    # Calculate the total weight of the tour and create a list of visited cities
    finalWeightChristofides = 0
    currentRouteCh = []
    visited = set()
    starting_vertex = chEulerianCircuit[0][0]
    currentRouteCh.append(starting_vertex)
    for finalEdges in chEulerianCircuit: 
        _, end = finalEdges
        if end not in visited: 
            currentRouteCh.append(end)
            visited.add(end)
        finalWeightChristofides += multiGraphGraph[finalEdges[0]][finalEdges[1]]['weight']
    
    # Return the tour and its total weight
    return currentRouteCh, finalWeightChristofides


def ant_colony_tsp(coords): 
    
    pheremoneVals = ones((len(coords), len(coords)))
    
    antDistanceVals = generate_distance_matrix(coords)
    
    bestPathVal = None
    
    bestDistVal = float('inf')
    
    iterationNumVal = max(100, len(coords)*5)
    
    antNumVal = max(10, len(coords)//2)
    
    #Above this initialize everything.
    
    for runs in range(iterationNumVal): 
        
        antCurPaths = []
        pathLenVals = []
        
        for ants in range(antNumVal): 
            
            curCityVal = random.randint(len(coords))
            
            visitCityList = [curCityVal]
            
            totDistTrav = 0
            #3 lines above intialize each ant's travels
            
            #This loop runs until all cities are visited
            while len(visitCityList) < len(coords): 
                
                notVisitedCityList = [place for place in range(len(coords)) if place not in visitCityList] #This line makes a list of the cities that have not been visited yet.
                
                nearCityVal = min(notVisitedCityList, key = lambda city: antDistanceVals[curCityVal][city])
                
                distNearCity = antDistanceVals[curCityVal][nearCityVal]
                #Accessing matrix to find nearest city and distance to nearest city.
                
                totDistTrav = totDistTrav + distNearCity
                
                visitCityList.append(nearCityVal)
                curCityVal = nearCityVal
                
            #return to start
            totDistTrav = antDistanceVals[curCityVal][visitCityList[0]]
            
            antCurPaths.append(visitCityList)
            
            pathLenVals.append(totDistTrav)
            
        pheromoneVals *= 0.9
        
        for path, distVal in zip(antCurPaths, pathLenVals): 
            
            for itemValues in range(len(path) - 1): 
                
                pheremoneVals[path[i]][path[i+1]] += 1/distVal
                pheremoneVals[path[i+1]][path[i]] += 1/distVal
                
    bestPathIndVal = argmin(pathLenVals)
    bestPathVal = antCurPaths[bestPathIndVal]
    bestDistVal = pathLenVals[bestPathIndVal]
    
    return bestPathVal, bestDistVal

def simulatedAnnealing(coords): 
    
    #Have to create the initial Matrix
    annealMatrix = generate_distance_matrix(coords)
    
    numberOfCities = len(coords) 
    
    #initialize cooling factor
    coolingFactor = 0.95
    
    #Create random path to start with 
    
    annealCurPath = list(range(numberOfCities))
    
    shuffle(annealCurPath)
    
    #Set initial temp very high
    
    annealTemp = 10000
    
    #set the initial best to the randomly generated set. (Calculate initial energy)
    
    bestAnnealPath = annealCurPath
    bestAnnealEnergy = calc_total_distance(annealCurPath, annealMatrix)
    
    for annealIterate in range(10000): 
        
        #generate new candidate solution
        
        newPathVal = annealCurPath.copy()
        
        shuffle(newPathVal)
        
        curEnergyVal = calc_total_distance(annealCurPath, annealMatrix)
        
        newEnergyVal =  calc_total_distance(newPathVal, annealMatrix)
        
        if newEnergyVal < curEnergyVal or uniform(0, 1) < exp((curEnergyVal - newEnergyVal)/annealTemp): 
            
            annealCurPath = newPathVal
            curEnergyVal = newEnergyVal
            
        if curEnergyVal < bestAnnealEnergy: 
            
            bestAnnealPath = annealCurPath
            bestAnnealEnergy = curEnergyVal
            
        annealTemp = annealTemp * coolingFactor
        #change the max temp
        
    return bestAnnealPath, bestAnnealEnergy

def run_algorithm(algorithm, coords):
    if algorithm == 'brute_force':
        return brute_force_tsp(coords)
    elif algorithm == 'nearest_neighbor':
         return nearest_neighbor_tsp(coords)
    elif algorithm == 'branch_and_bound':
         return branch_and_bound_tsp(coords)
    elif algorithm == 'held_karp':
        return held_karp_tsp(coords)
    elif algorithm == 'two_opt':
        return two_opt_tsp(coords)
    elif algorithm == 'christofides':
        return christofides_tsp(coords)
    elif algorithm == 'ant_colony':
        return ant_colony_tsp(coords)
    elif algorithm == 'simulated_annealing':
        return simulatedAnnealing(coords)

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Solve TSP problem using different algorithms.')
    parser.add_argument('dataset', type=str, help='TSP dataset file path.')
    parser.add_argument('algorithm', type=str, choices=['brute_force', 'nearest_neighbor', 'branch_and_bound', 'held_karp', 'two_opt', 'christofides', 'ant_colony', 'simulated_annealing'], help='Algorithm to use.')
    parser.add_argument('--timeout', type=int, default=60, help='Timeout for the algorithm in seconds.')
    args = parser.parse_args()

    dataset_filename = os.path.basename(args.dataset) # Extract filename from the dataset path
    
    # Read dataset and execute algorithm
    coords = read_tsp_file_from_disk(args.dataset)  # <-- Modified this line

    start_time = time.time()

    try:
        path, distance = func_timeout(args.timeout, run_algorithm, args=(args.algorithm, coords))
    except FunctionTimedOut:
        print(f"{args.algorithm.replace('_', ' ').title()} took too long to run for {dataset_filename} with {len(coords)} cities. Skipping...")
        return  # Exiting the function as the algorithm didn't complete within the timeout

    totalTime = time.time() - start_time

    if path is not None and distance is not None:
        print(f"Dataset: {dataset_filename} | Algorithm: {args.algorithm.replace('_', ' ').title()} | Path: {path} | Number of locations: {len(coords)} | Distance: {distance} | Elapsed time: {totalTime} seconds")
        
        # Plot the path
        plot_path(coords, path, args.algorithm, distance, totalTime)


# Entry point of the script
if __name__ == "__main__":
    main()
