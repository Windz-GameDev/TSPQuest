'''
    Source Code for Algorithms Project for COT6405

    Algorithms for Brute Force, and Nearest Neighor were initially 
    referenced from GeeksForGeeks to ensure accurate implementation. Branch and bround is
    referenced from Abdul Bari's guide on youtube linked below. Dynamic Programming solution 
    used multiple resources as a reference. Datasets are either auto generated or sourced from
    The University of Waterloo in Waterloo, Ontario, Canada
    2-opt is referenced from KeiruaProd to ensure accurate implementation.

    References: 
    BruteForce 
      1. https://www.geeksforgeeks.org/traveling-salesman-problem-tsp-implementation/
    Nearest Neighbor 
      1. https://www.geeksforgeeks.org/travelling-salesman-problem-greedy-approach/
    Branch and Bound 
      1. https://www.youtube.com/watch?v=1FEP_sNb62k&t=515s
      2. https://www.geeksforgeeks.org/traveling-salesman-problem-using-branch-and-bound-2/
    Dynamic Programming 
      1. https://www.geeksforgeeks.org/travelling-salesman-problem-using-dynamic-programming/
      2. https://www.youtube.com/watch?v=Q4zHb-Swzro&t=0s
      3. https://www.youtube.com/watch?v=JE0JE8ce1V0&t=705s
    2-Opt - 
      1. https://www.keiruaprod.fr/blog/2021/09/15/traveling-salesman-with-2-opt.html
    Christofides - 
      1. https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/
      2. https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.operators.binary.compose.html
      3. http://matejgazda.com/christofides-algorithm-in-python/
      4. https://www.youtube.com/watch?v=Uu2ptesvteE
      5. https://notebook.community/DhashS/Olin-Complexity-Final-Project/code/03_approximation_algorithms
      6. https://networkx.org/documentation/stable/reference/convert.html
      7. https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.approximation.traveling_salesman.christofides.html#christofides
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
from networkx.algorithms import approximation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import *


def plot_path(coords, path, algorithm, total_distance, time, dataset_name):

    # Set figure size (width, height in inches)
    plt.figure(figsize=(10, 8))

    # Unpack the coordinates list of tuples into two seperate list, x, and y, zip groups first elements (x-coordinates together, and second elements (y-coordinates together)
    x, y = zip(*coords)

    # Create a scatter plot of the cities, takes two arguments, x coordinates, and y-coordinates and plots these points on a 2D graph.
    plt.scatter(x,y)

    # Plot the path taken
    for i in range(len(path) - 1): # Stop at one less than the final city because final city will try to connect to a city out of bounds
        plt.plot((coords[path[i]][0], coords[path[i+1]][0]), (coords[path[i]][1], coords[path[i+1]][1]), 'r-') # First argument is a tuple containing the x coordinates of the first and second city, the second tuple is for the y coordinates respectively
    # plt.plot((coords[path[-1]][0], coords[path[0]][0]), (coords[path[-1]][1], coords[path[0]][1]), 'r-') # Connect the final city in the path to the first city in the path incase the first city was not appended to the end of the path list

    # Label the starting node as "Start"
    start_x, start_y = coords[path[0]]
    plt.text(start_x, start_y, 'Start', fontsize=10, ha='right', weight='bold')

    # Add title
    plt.title(f"{algorithm.replace('_', ' ').title()} - Cities: {len(coords)} - Distance (Euclidean): {round(total_distance, 4)} - Time (Seconds): {time}")

    # Add labels
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")

    # All figures go in the results directory
    results_dir = 'path_results'
    
    # Ensure the main results directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create a subdirectory for the dataset within the results directory
    dataset_dir = os.path.join(results_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Initial number to append to filename
    counter = 1

    # Check for existing files and increment counter
    while True:
        file_path = os.path.join(dataset_dir, f"{algorithm}_{counter}.png")
        if not os.path.exists(file_path):
            break
        counter += 1

    # Save the plot with the incremented filename
    plt.savefig(file_path)
    print(f"Plot saved as {file_path}")
    plt.close()

'''
def example_graph(coords, dataset_name):
    # Set figure size (width, height in inches)
    plt.figure(figsize=(10, 8))

    # Unpack the coordinates list of tuples into two separate lists, x, and y
    x, y = zip(*coords)

    # Create a scatter plot of the cities
    plt.scatter(x, y)

    # Label each point with its index
    for i in range(len(coords)):
        plt.text(coords[i][0], coords[i][1], f"{i}", fontsize=10, ha='right', weight='bold')

    # Add title and axis labels
    plt.title(f"Example for {dataset_name}")
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")

    # Ensure the results directory and dataset subdirectory exist
    results_dir = 'path_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    dataset_dir = os.path.join(results_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Save the plot
    file_path = os.path.join(dataset_dir, "Example.png")
    plt.savefig(file_path)
    print(f"Plot saved as {file_path}")
    plt.close()
'''
    
'''
    Generate a euclidean distance matrix for a list of coordinate tuples.

    The self-distance for every city to itself should be infinity for Branch and Bound to avoid an algorithm considering loops which return to the same city.
    However, it is 0 by default for other algorithms.

    The first nested for loop's inner loop creates a row equal in length to the number of cities, and the outer loop causes this loop
    to execute a number of times equal to the number of cities. This results in a matrix with an equal number of 
    rows and columns.

    The second nested for loop calculates the euclidean distance between all city pairs, and stores them in a matrix. The outer loop is responsible for
    navigating the different rows, and the inner for loop for navigating the columns of each row. The row represents the starting city, and the column
    the destination city.

    There are two nested for loops so this function takes 2 * N^2 time, simplifying to O(N^2) in big O notation. Given an input of size N, it takes N * N 
    storage leading to a space complexity of O(N^2).
'''
def generate_distance_matrix(coords, self_distance = 0):
    num_cities = len(coords)
    matrix = [[self_distance for city in range(num_cities)] for city in range(num_cities)] 
    
    # Calculate the distance for each city i to every other city
    for i in range(num_cities):
        for j in range(num_cities): 
            # Any city traveling to itself should have a distance of 0 or infinity depending on the algorithm, for example city 0 -> city 0 results in cost of 0 or infinity
            if i != j: 
                # Calculate the euclidean distance for each city pair, and use them to populate the distance matrix
                matrix[i][j] = euclidean_distance(coords[i], coords[j]) 
    return matrix

'''
    Performs row and column reduction on a matrix and calculates the reduction cost. Used in the approximation of a lower bound of a node
    in the Branch and Bound Algorithm. 

    We subtract the minimum of each row, from each element in that row, adding each minimum found to our reduction cost as long as that minimum is not 0 or infinity.

    Afterwards, we do the same thing, except for columns.

    O(n^2) time complexity due to the two nested for loops.
'''
def reduce_matrix(distance_matrix):
    num_cities = len(distance_matrix)
    cost = 0 # Initialize cost of reduction 0 

    # Reduce rows
    for row in range(num_cities):
        min_value = min(distance_matrix[row]) # Find the minimum value in the row
        
        # If min value is 0, the row is already reduced. If there are only infinities, we do not want to add infinity to the cost!
        if min_value != 0 and min_value != float('inf'):
            cost += min_value # Add minimum value from each row to the cost
            for column in range(num_cities):
                distance_matrix[row][column] -= min_value # Subtract the minimum value from every element in the row

    # Reduce columns
    for column in range(num_cities): # Iterate through each column 
        min_value = min(distance_matrix[row][column] for row in range(num_cities)) # Get the minimum value in the column
        
        # If min value is 0, the column is already reduced. If there are only infinities, we do not want to add infinity to the cost!
        if min_value != 0 and min_value != float('inf'): 
            cost += min_value # Add minimum value from each column to the cost
            for row in range(num_cities):
                distance_matrix[row][column] -= min_value # Subtract the min value from each row in the column

    # Return reduced distance matrix, aditionally return the cost to be used in calculating the lower bound of a node 
    return distance_matrix, cost 

'''
    The euclidean_distance function takes two coordinate tuples representing cities.
    All coordinate tuples contain a city's x and y coordinates in the TSP problem, indexed 0 and 1
    respectively. The first tuple represents the current city, and the second tuple is the destination city.

    The function computes the euclidean distance by calculating the square root of the sum of the squared 
    differences in the x and y coordinates of the two points.

    When the function completes, it returns the euclidean distance between these two cities. 
    This a measure of the straight-line distance between them in a 2D plane.

    This function runs in O(1) or constant time, as the computation requires 
    a fixed number of arithmetic operations, regardless of input size.
'''
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

#This calculates total distance for the simulated annealing function
def calc_total_distance(pathTotVal, distTotMatrix): 
    
    sumTotDist = 0
    
    for numValTot in range(len(pathTotVal)-1): 
        
        sumTotDist += distTotMatrix[pathTotVal[numValTot]][pathTotVal[numValTot+1]]
        
    sumTotDist += distTotMatrix[pathTotVal[-1]][pathTotVal[0]]
    
    return sumTotDist
    

'''
    Simplest solution for solving the TSP, but also the longest. The brute force approach generates all possible path permutations which visit each city exactly one time,  
    except for the starting city at the beginning and again at the end which is required to form a Hamiltonian Cycle. A Hamiltonian cycle in the TSP context 
    refers to a complete tour that visits each city eaxctly once and returns to the starting city. Generating each permutation has a big O(n!) time complexity. After generating each permutation it  
    checks the length of each path, and returns the smallest one. This garuntees this algorithm will return the best possible answer. 
    It does this keeping track of the min distance and best path, and updating these variables when a new better path is found. This path checking process is O(n) for each permutation, making the 
    total time complexity O(n! * n). The storage requirement is O(n), this is because aside from the input, only the best path and the current path are stored at any specific moment in time. 
    This is due the memory efficiency of the iteertools permutation function. Since both both paths are the size of N, this would be 2N, in Big O Notation we may drop the constant, so this simplifies to O(N).
'''
def brute_force_tsp(coords): # Coords represents a list of tuples, the 0th element of any tuple represents the x element, and the 1st element of any tuple represents the y element
    min_distance = float('inf')  # Keep track of shortest distance so far, start at infinity so all paths found will be less than initial value
    best_path = None  # Keep track of the coordinate indices that represent the best path found so far
    
    ''' Take the number of coordinates, generate a range of numbers from that length, each representing a coordinate index, 
    going from 0 to len(coords) - 1, and generate all permutations of that range, representing all paths we can take through the list of coordinates'''
    for current_path in permutations(range(len(coords))): # Calculate the distance of each permutation or path
        current_distance = 0 # Start with a distance of 0 for each path or permutation
        min_potential = True
        for i in range(len(current_path) - 1): # We need to start stop before the final location in the path because we access the next index in the loop.
            ''' We calculate the euclidean distance between two coordinates by calling the euclidean_distance function, 
            We pass in two coordinate tuples from the coordinates list, accessing them using the path indices of the current location and the next in the path'''
            current_distance += euclidean_distance(coords[current_path[i]], coords[current_path[i+1]]) 
            
            # If current permutation clearly will not be better than the current min, we can break out early
            if (current_distance > min_distance):
                min_potential = False
                break

        # If loop was broken out of early, reason to continue, move on to next permutation
        if (min_potential == False):
            continue

        # Now, we connect the final location in the path to the starting position to complete the cycle
        current_distance += euclidean_distance(coords[current_path[-1]], coords[current_path[0]]) 

        # Update min_distance and best_path
        if current_distance < min_distance: # If the current path distance just calculated is less than the stored minimum we've found so far, we have a new minimum and best path
            min_distance = current_distance # Assign the new minimum euclidean distance to minimum distance
            best_path = list(current_path) # Assign a list copy of the new best path permutation tuple to best path, the indices representing the coordinates locations in the coords list of tuples 
    
    best_path.append(best_path[0]) # Make the first element of the best path the last element as well to complete the hamiltonian cycle
    return best_path, min_distance # Return the optimal answer to the TSP problem 

'''
    The below function implements the greedy approach for the TSP, also known as nearest neighbor. This algorithm works by making the locally best decision at any point in time.
    However, this approach may not return the best or optimal decision like brute force. The advantage of this algorithm is that it runs in Polynomial time O(n^2) vs O(n! * n) of Brute Force. 
    In addition it also uses 2 * N or O(n) storage space.
'''
def nearest_neighbor_tsp(coords):

    # The number of coordinate point tuples in the coords list gives us the number of cities in the problem
    num_cities = len(coords) 
    
    # Keep track of the unvisted cities using a set of integers, each representing the index of a point in the coords list from 0 to len of num_cities - 1. This takes O(n) space.
    cities_left_to_visit = set(range(num_cities)) 

    # Initialize an empty list which will contain the path of city indexes for the coords list (will store found path by NN). Takes O(n) space.
    found_path = [] 

     # Start from the first city in the dataset (arbitrarily chosen)
    current_city = 0 

    # Add first city to the found path so we can keep track of our starting point
    found_path.append(current_city) 

    # First city has been visited so it should not be visited again until the very end of the algorithm to complete the hamiltonian cycle
    cities_left_to_visit.remove(current_city) 

    # Keep track of the total distance of the found path, we start at 0 since we are still at the starting position
    found_distance = 0 
    
    '''
        Visit nearest unvisited city as long as all cities have not been visited. The time complexity of the code inside the while loop is 
        O(n). Since the while loop will iterate n - 1 times, the while loop contributes a time complexity of O(n^2).
    '''
    while cities_left_to_visit:
        '''
            To find where we need to go from the current city, we need to look through all of the unvisited cities, and find the one that adds the minimum euclidean distance
            to our path in the short term. We do this passing the cities_left_to_visit set to the min function, and give each index in the set a key which is considered by the min 
            function in its comparisons. We do this by using a key argument, a function to compute the key for each city index, taking the currently considered city in the set as a
            parameter. The computed key is the euclidean distance from the current city to the next city being considered. We get the euclidean distance by passing in the indexes of 
            the starting city and the destination city being considered in the unvisited set to the euclidean distance function as arguments. The city index in the unvisited set
            with the smallest key will be returned by the min function. This process of finding the minimum euclidean distance for each move is O(n). Only two keys are ever stored at a
            time in the minimum function, the minimum key, and the key of the city being evaluated to see if it's less than the minimum. Therefore the keys contribute a storage complexity of O(1).
        '''
        nearest_city = min(cities_left_to_visit, key=lambda city: euclidean_distance(coords[current_city], coords[city])) 

        # Since we are traveling to next city in the path, we have to add the distance to get there to total distance
        found_distance += euclidean_distance(coords[current_city], coords[nearest_city]) 

        # Adjust our current location to the city we just traveled to
        current_city = nearest_city 

        # Add the index of the current city to our found path so we know the path which provided our found distance at the end
        found_path.append(current_city) 

        # Our current city has been visited so we should not travel to it again according to the definition of the TSP problem and a Hamiltonian cycle.
        cities_left_to_visit.remove(current_city)
    
    # Now that every city has been visited, we must return from the last city back to the start to complete the hamiltonian cycle
    found_distance += euclidean_distance(coords[current_city], coords[found_path[0]]) 
    found_path.append(found_path[0])

    # Now that we have the path and total distance, we must return them so we can visualize the results
    return found_path, found_distance 

'''
    Branch and Bound algorithm for solving the TSP. It uses an optimized search strategy to systematically divide and conquer the solution space. 
    It works by maintaining an upperbound and a priority queue. The upperbound is retrieved from Nearest Neighbor to provide us with a decent starting point which
    can used to quickly prune unfeasible paths. The priority queue begins with the starting node being enqueued. After, we enter a while loop, where as long as we have nodes 
    in the priority queue, we continue the while loop. From there, the starting node is poppped, its unvisited neighors are checked, and an estimated cost or lower bound 
    is calculated for each of them. Each of these unvisited neighbors represent a partial solution to the problem, with one more city in their solution, one of the previous
    node's unvisited neighbors.

    Any unvisited node whose lowerbound is less than the upperbound is enqueued to the priority queue to later be expanded, as long as they are less than the new 
    upperbound if one is found in the time between when the node was enqueued and when it is popped from the priority queue and checked to see if it should be expanded. 
    After all of a node's unvisited node's have been checked, we return to the priority queue, popping the next node with the lowest cost, and seeing if its cost is lower 
    than the upperbound. If it is, we expand it, and enqueue its neighbors if their estimated solution cost is lower than the upperbound. This process repeats until
    a leaf node is reached, in which case, if its cost is less than the upperbound, we update the upperbound distance and path to those of the current node.

    This process repeats until all potential nodes or potential paths are pruned from the priority queue. The heuristic for estimating a node's lower bound uses a matrix reduction, 
    the cost to reach the previous node before the node being checked, and the distance from the previous node's last city to next city represented by this node. This solution
    guarantees optimality and is superior to Brute Force in the terms of its practical run time. 
    
    The runtime can be affected by the effectiveness of the given upperbound and the problem dataset itself. That being said, the theoretical worst case time of complexity 
    of this implmentation can be approximately bounded by O(n! * n^2). This is because in the worst case scenario, we are unable to prune any paths, and we explore every permutation. Each node represents a partial 
    solution or permutation, storing a distinct distance matrix, cost, path, and a set of unvisited neighbors for that solution so far.
    In addition, each node takes O(n^2) time complexity due to its matrix operations, however this is very unlikely for the vast majority of datasets. 

    We can give this algorithm implementation a space complexity of O(M * n^2) where M is representative of the maximum number of nodes in the priority queue at any 
    given time. Each node is O(n^2) storage because it stores a distance matrix for the lowerbound heuristic.
    M is dificult to quantify in terms of N and can vary depending on the efficiency of pruning, however M is significantly less than n! as not all permutations 
    or partial permutations will be stored in the priority queue at the same time. 
    
    It is very unlikely this algorithm will approach these mentioned worst case space and time complexities in practice, as it is almost some certain some pruning will occur during the algorithm. However, the 
    effectiveness of this pruning will vary based on the dataset the algorithm was run on.
'''
def branch_and_bound_tsp(coords):
    
    # Generate the distance matrix from the list of coordinate tuples
    distance_matrix = generate_distance_matrix(coords, float('inf'))
    
    '''
        Define a node class for our state based tree which will store a node's path, it's reduced matrix, and its cost.

        The space complexity of each node is O(n^2), this is because of the necessity to store the distance matrix.
    '''
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
 
    # Get an upperbound and starting path using NN. All node costs must less than upperbound to avoid being pruned. This uses O(n^2) time, and O(n) space.
    best_path, upper_bound = nearest_neighbor_tsp(coords) 

    distance_matrix, starting_cost = reduce_matrix(distance_matrix) # O(n^2) time and space
    starting_node = Node(distance_matrix, starting_cost, [0], len(coords), set(range(len(coords))))
    starting_node.unvisited.remove(0) # Remove starting the city from the starting node's set of unvisited neighbors, this ensures the starting node is not returned to until the end.
    priority_queue = [starting_node] # Our priority queue ensures we always explore the node with the lowest cost first, and at first containins only the root

    # While not all paths have been pruned 
    while priority_queue:
        current_node = heapq.heappop(priority_queue) # O(logn)

        # If the cost of the current node is greater than the upper bound, skip this node, upperbound may have changed while nodes were waiting in queue
        if current_node.cost > upper_bound:
            continue

        if (current_node.is_complete_tour):

            # Node is a complete tour and cost is less than or equal to upper bound, so update upperbound
            upper_bound = current_node.cost
            best_path = current_node.path + [0] # Add the starting city to complete the new best path
            continue

        for unvisited in current_node.unvisited: 
            child_matrix = deepcopy(current_node.matrix) # Start with a copy of parent matrix. O(n^2) Time and Space.
            
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

            # Reduce child matrix. O(n^2) time complexity.
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

    # Used to verify BB calculates the correct total distance, including distance back to the start node
    '''
        total_distance = 0
        for i in range(len(best_path) - 1):
            city = best_path[i]
            next_city = best_path[i + 1]
            total_distance += euclidean_distance(coords[city], coords[next_city]) 
        print(total_distance)
    '''

    return best_path, upper_bound # Return the best path and minimum cost 

'''
    Dynamic programming algorithm for solving the TSP 

    It uses recursion to find the optimal cost given a city and a subset of remaining cities. Whether or not a city is visited in a subset is represented 
    by a bitmask. Essentially the ith city, starting at index 0, in a binary number is set to 1 to represent a visited city, and 0  for one not yet visited.

    The time complexity is O(n^2 * 2^n). Theoretically we may call the recursion function for nearly each combination of n cities and 2^n subsets. Since each individual 
    call of the recursion function is O(n) to check for unvisited cities, we can determine the previously mentioned time complexity of O(n^2 * 2^n). 

    The space complexity is O(n * 2^n) as this is required for the memoization table to store all possible subsets for n cities which may be reached 
    during the recursion process.

    Overall, the held karp algorithm improves the run time of the brute force algorithm by using smart recursion, using increased storage, and reusing solutions 
    between cities and subsets of visited cities.

'''
def held_karp_tsp(coords):    
    '''
        This function is a key component of the Held Karp Algorithm. It recursively computes the minimum cost of completing a tour, starting from a given city
        and visting a specific subset of remaining cities. Where the problem is distinguished from brute force is that in the case of reappearing subproblems, 
        the answer to that subproblem is retrieved from a memoization table to avoid repeat calculations. If the subproblem was not previously calculated, it is stored in 
        the memoization table for future use.
    '''
    def recursion(currently_visited, city):
        # Base case, reached end city in the path, calculate the distance from the final node, to the starting node
        if(currently_visited == all_cities_visited):
            # Cost of the current city to an empty unvisited subset is the distance from the city to the starting city 
            memo[city][currently_visited] = (distance_matrix[city][0], 0);
            return memo[city][currently_visited][0] # Return cost stored in the memo table

        # If the min cost from this city to the next optimal city in this subset of unvisited cities has already been calculated, return it from the memoization table 
        if (memo[city][currently_visited][0] != float('inf')):
            return memo[city][currently_visited][0] 

        # For each point in the path start, with a min cost of infinity, update everytime you find a next city that leads to a shorter cost
        min_cost = float('inf')
        min_city = None

        # Check for unvisited cities, O(n) time and space
        for next_city in range(num_cities):
            '''
                1 << next_city will create a binary number with only the bit representing the integer index value held by next_city, set to 1
                
                memo[city] == gives a bit mask which tells 

                currently_visited & (i << next_city) performs a bitwise AND between the currently visited bit_mask and the
                the binary number with only the next city's index set to 1.

                The above bitwise AND operation creates a new binary number, where there can only be one bit set 
                to 1. That is, if the number held by next_city's bit that was set to one by 1 << next_city is also set to 1 in currently_visited, 
                then next_city has already been visited in the currently_visited subset. Here is a visual example:

                
                00101 = currently visited's bitmask 
                00001 = 1 << next_city where next_city = 0

                00101 & 00001 = 000001
                Since 000001 != 0, then we know that next_city's bit is set to 1 in the currently_visited subset, therefore, we know the 0th city has been visited.

                Take the counter example

                00101 = currently visited
                00010 = 1 << next_city where next_city = 1

                00101 & 00010 = 00000,
                Since 00000 == 0, we know that the 1st city has not visited, so we should explore it. 


                In short, if the binary number as a result of the bitwise AND
                is 0, the number held by next_city's bit in the currently_visited bitmask is 0. This means next city
                in the range num_cities has not been visited yet, meaning we should explore it. 
            '''
            if ((currently_visited & (1 << next_city)) == 0):
                '''
                    Get the cost of this city to the next unvisited one, and the cost of the rest of the optimal decisions down this path

                    When we are passing the new bit mask represent the new visited cities for the next city. We simply set the bit at 
                    index next_city to 1 by using a bitwise OR operation. A bit wise OR operation traverses each bit in both binary numbers,
                    and if either bit in a position is one, the resulting binary number is set to 1 for that bit position. 

                    See the below example:

                    00101 = currently visited
                    00010 = 1 << next_city where next_city = 1

                    00101 OR 00010 = 00111

                    The binary 00111 binary number means that the 0th, 1st, and 2nd city have all been visited. This is important for making sure a city does 
                    not revisit itself.  

                    next_city is merely a city's coordinate index in the coordinate list of tuples
                '''
                next_city_cost = distance_matrix[city][next_city] + recursion(currently_visited | (1 << next_city), next_city)
                
                # Cost for taking this city next from the previous city is less than the currently found minimum cost, so update it
                if (next_city_cost < min_cost):
                    min_cost = next_city_cost
                    min_city = next_city
                    # Store the min cost so far from this city to every city in the set of currently unvisited cities, as well as the next optimal city

        # No more unvisited cities, we have the minimum cost for this city to this subset so we return it up to the next level
        memo[city][currently_visited] = (min_cost, min_city)
        return memo[city][currently_visited][0]
    
    '''
        This function will allow us to reconstruct the optimal path given an already filled out memoization table from the recursion algorithm.
        From the start of the optimal path in the memoization table, the first city to a subset where the first city is the only one visited, 
        it continously accesses the next node which was used to provide the optimal path cost.

        This function takes O(n) time, this is because each cell in the memoization table 
        will allow the program to the add the next optimal city in the path to the best_path variable n times.
    '''
    def reconstruct_path():
        # Bitmask where first city (0) is visited and all others are unvisited
        currently_visited = 1 << 0
        
        best_path = [0] # We always start from city 0
        next_city = 0 # Initialize with starting city
        
        # While not all cities have been visited, or the bit mask does not have all bits set to 1 for each city
        while currently_visited != (1 << num_cities) - 1:
            min_cost, next_city = memo[best_path[-1]][currently_visited]
            
            # For troubleshooting
            if next_city is None: 
                raise ValueError(f'No next city found for city {best_path[-1]} and visited cities {currently_visited}')
            
            # Add the next optimal city to the best path that we just found from the memo table
            best_path.append(next_city)
            
            # Mark the next city as visited in the bit mask
            currently_visited = currently_visited | (1 << next_city) # 0 or 1 is always 1
        
        # Add the starting city to complete the hamiltonian cycle
        best_path.append(0)
        return best_path
    
    # Obtain the number of cities which will be used to help create our distance matrix and memoization table
    num_cities = len(coords)

    # Create a distance matrix to quickly obtain distances between cities O(n^2) time and storage
    distance_matrix = generate_distance_matrix(coords, 0)

    ''' 
        Create a 2D list with 2^n columns and n rows where n represents the number of ciites.
        Each row corresponds to a city, and each column corresponds to a subset of cities.

        Each cell contains a tuple holding the minimum cost to visit all cities in a subset, 
        and the next city in the optimal path which is necessary for back tracking.

        Ex: 
            
        memo[i][j][0] represents the minimum cost to travel from a city i, to visit all cities
        in a subset j.

        O(2^n * n) time and storage

    '''
    memo = [[(float('inf'), None) for _ in range(1 << num_cities)] for _ in range(num_cities)]
    '''
        A left shift operation that shifts the binary representation of the number 1 to the left by num_cities places,.
        creating a binary number with a number of bits equal to the number of cities + 1, where each bit except 
        the first is set to 0 except the last. If there five cities, this would produce 100000. Which is equal to 2^5 or 32 in binary. 
        Then, it subtracts 1 from this binary number to give us 11111 to give us a binary number with a number of bits equal to the
        number of cities, or 31. 
        
    '''
    all_cities_visited = (1 << num_cities) - 1 
    
    # Create a binary number holding 1, this represents our starting point in the algorithm, the 0th city
    track_visited_cities = 1 << 0 

    # Fill out the memoization table, and retrieve the final cost of the optimal path
    final_cost = recursion(track_visited_cities, 0)

    # Back track through the table to obtain the path which provided the final_cost
    final_path = reconstruct_path()
    return final_path, final_cost
      
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
    
    if currentRoute[0] != 0: 
        
        currentDist += euclidean_distance(coords[0], coords[currentRoute[0]])
        
        currentRoute.insert(0, 0)
        
    if currentRoute[-1] != 0: 
        
        currentDist += euclidean_distance(coords[currentRoute[0]], coords[currentRoute[-1]])
        
        currentRoute.append(0)
        
    
    return currentRoute, currentDist

def christofides_tsp2(coords):
    '''
        This function implements the Christofides algorithm for the Traveling Salesman Problem (TSP).
        It takes as input a list of coordinates and returns a tour and its total weight.
        
        Dataset must form a metric space
        
        Utilizes networkx built in christofides function, other function 'çhristofides_tsp' does not
          https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.approximation.traveling_salesman.christofides.html#christofides
          https://networkx.org/documentation/stable/_modules/networkx/algorithms/approximation/traveling_salesman.html
    '''

    # Generate the distance matrix for the given coordinates
    christofidesMatrix = generate_distance_matrix(coords)
    
    # Does the dataset fail the symmetric requirement for forming a metric space?
    triangleSymmetricTruth = False
    
    for i in range(len(christofidesMatrix)):
        if triangleSymmetricTruth == True:
            break
        for j in range(len(christofidesMatrix)):
            if christofidesMatrix[i][j] != christofidesMatrix[j][i]:
                triangleSymmetricTruth = True
                break

    # Warn the user if the dataset fails the symmetric requirement for forming a metric space
    if triangleSymmetricTruth == True:    
        print("The distances are not symmetric.")

    # Does the dataset fail the triangle inequality requirement for forming a metric space?
    triangleInequalityTruth = False
    
    for i in range(len(christofidesMatrix)):
        if triangleInequalityTruth == True: 
            break
        for j in range(len(christofidesMatrix)):
            if triangleInequalityTruth == True: 
                break
            for k in range(len(christofidesMatrix)):
                if christofidesMatrix[i][k] > christofidesMatrix[i][j] + christofidesMatrix[j][k]:
                    triangleInequalityTruth = True
                    break

    # Warn user if dataset doesn't meet triangle inequality requirement                
    if triangleInequalityTruth == True: 
        print("The distances do not obey the triangle inequality.")
        

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
        node1, node2 = sorted(edge)
        mst[node1][node2]['weight'] = christofidesMatrix[node1][node2]

    # Find the approximate shortest path using Christofides algorithm through the networkx library
    shortest_path = approximation.christofides(nxGraph, weight='weight', tree=mst)
    print(shortest_path)
    
    # Calculate the total distance of the tour
    total_distance = 0
    for city in range(len(shortest_path) - 1): 
        total_distance += christofidesMatrix[shortest_path[city]][shortest_path[city + 1]]
    total_distance += christofidesMatrix[shortest_path[-1]][shortest_path[0]]

    return shortest_path, total_distance


"""

This function implements the Christofides algorithm for the Traveling Salesman Problem (TSP).
It takes as input a list of coordinates and returns a tour and its total weight.

Dataset must form a metric space

Implemented without networkx built in function, used by default

"""

def christofides_tsp(coords):
    # Generate the distance matrix for the given coordinates
    christofidesMatrix = generate_distance_matrix(coords)
    
    # Does the dataset fail the symmetric requirement for forming a metric space?
    triangleSymmetricTruth = False
    
    for i in range(len(christofidesMatrix)):
        if triangleSymmetricTruth == True:
            break
        for j in range(len(christofidesMatrix)):
            if christofidesMatrix[i][j] != christofidesMatrix[j][i]:
                triangleSymmetricTruth = True
                break

    # Warn the user if the dataset fails the symmetric requirement for forming a metric space
    if triangleSymmetricTruth == True:    
        print("The distances are not symmetric.")

    # Does the dataset fail the triangle inequality requirement for forming a metric space?
    triangleInequalityTruth = False
    
    for i in range(len(christofidesMatrix)):
        if triangleInequalityTruth == True: 
            break
        for j in range(len(christofidesMatrix)):
            if triangleInequalityTruth == True: 
                break
            for k in range(len(christofidesMatrix)):
                if christofidesMatrix[i][k] > christofidesMatrix[i][j] + christofidesMatrix[j][k]:
                    triangleInequalityTruth = True
                    break

    # Warn user if dataset doesn't meet triangle inequality requirement                
    if triangleInequalityTruth == True: 
        print("The distances do not obey the triangle inequality.")

    # Initialize a new NetworkX graph
    nxGraph = networkx.Graph()
    nxGraph.add_nodes_from(range(len(coords)))

    # Add all the edges to the graph, with their weights set to the corresponding distances in christofidesMatrix
    for nxItemOne in range(len(coords)): 
        for nxItemTwo in range(nxItemOne + 1, len(coords)):    
            nxGraph.add_edge(nxItemOne, nxItemTwo, weight=christofidesMatrix[nxItemOne][nxItemTwo])

    # Compute the Minimum Spanning Tree (MST) of the graph
    mst = networkx.minimum_spanning_tree(nxGraph)

    # Add weights to the edges of the MST
    for edge in mst.edges():
        node1, node2 = sorted(edge)
        mst[node1][node2]['weight'] = christofidesMatrix[node1][node2]

    minWeightGraph = nxGraph.copy()
    minWeightGraph.remove_nodes_from([v for v, degree in mst.degree() if not (degree % 2)])
    edges = networkx.algorithms.min_weight_matching(minWeightGraph)


    multiGraphGraph = networkx.MultiGraph()
    multiGraphGraph.add_edges_from(mst.edges(data=True))
    multiGraphGraph.add_edges_from(edges)

    # Check if the graph is connected
    if not networkx.is_connected(multiGraphGraph):
        raise networkx.NetworkXError("Graph is not connected.")

    # Check if the graph is Eulerian
    if not networkx.is_eulerian(multiGraphGraph):
        raise networkx.NetworkXError("Graph is not Eulerian.")

    # Find an Eulerian circuit in the graph
    chEulerianCircuit = list(networkx.eulerian_circuit(multiGraphGraph))
   
    # Initialize a list to store the TSP tour
    found_path = []
    visited = set()

    # Iterate through the Eulerian circuit
    for edge in chEulerianCircuit:
        start, end = edge

        # Add the starting node to the tour if it hasn't been visited
        if start not in visited:
            found_path.append(start)
            visited.add(start)

        # Add the ending node to the tour if it hasn't been visited
        if end not in visited:
            found_path.append(end)
            visited.add(end)

    # Add the starting node to close the TSP tour
    found_path.append(found_path[0])

    # Calculate the total distance of the tour
    total_distance = 0
    for i in range(len(found_path) - 1):
        total_distance += christofidesMatrix[found_path[i]][found_path[i + 1]]
    
    # Return the tour and its total weight
    return found_path, total_distance

def ant_colony_tsp(coords): 
    
    pheremoneVals = ones((len(coords), len(coords)))
    
    antDistanceVals = generate_distance_matrix(coords)
    
    bestPathVal = []
    
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
            totDistTrav += antDistanceVals[curCityVal][visitCityList[0]]
            
            antCurPaths.append(visitCityList)
            
            pathLenVals.append(totDistTrav)
            
        pheremoneVals *= 0.9
        
        for path, distVal in zip(antCurPaths, pathLenVals): 
            
            for itemValues in range(len(path) - 1): 
                
                pheremoneVals[path[itemValues]][path[itemValues+1]] += 1/distVal
                pheremoneVals[path[itemValues+1]][path[itemValues]] += 1/distVal
                
    bestPathIndVal = argmin(pathLenVals)
    bestPathVal = antCurPaths[bestPathIndVal]
    bestDistVal = pathLenVals[bestPathIndVal]
    
    if bestPathVal[0] != 0: 
        
        #bestDistVal += euclidean_distance(coords[0], coords[bestPathVal[0]])
        
        bestPathVal.insert(0, 0)
        
    if bestPathVal[-1] != 0: 
        
        #bestDistVal += euclidean_distance(coords[bestPathVal[-1]], coords[0])
        
        bestPathVal.append(0)
        
    
    return bestPathVal, bestDistVal

def simulatedAnnealing(coords): 
    
    #Have to create the initial Matrix
    annealMatrix = generate_distance_matrix(coords)
    
    numberOfCities = len(coords) 
    
    #initialize cooling factor
    coolingFactor = 0.95
    
    #Create random path to start with 
    
    annealCurPath = list(range(1, numberOfCities))
    
    shuffle(annealCurPath)
    
    annealCurPath = [0] + annealCurPath + [0]
    
    #Set initial temp very high
    
    annealTemp = 10000
    
    #set the initial best to the randomly generated set. (Calculate initial energy)
    
    bestAnnealPath = annealCurPath
    bestAnnealEnergy = calc_total_distance(annealCurPath, annealMatrix)
    
    for annealIterate in range(10000): 
        
        #generate new candidate solution
        
        restOfPath = annealCurPath[1:-1].copy()
        
        shuffle(restOfPath)
        
        newPathVal = [0] + restOfPath + [0]
        
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
    
    if bestAnnealPath[0] != 0: 
        
        bestAnnealPath.insert(0, 0)
        
    if bestAnnealPath[-1] != 0:
        
        bestAnnealPath.append(0)
    
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
    # example_graph(coords, dataset_filename)

    start_time = time.time()

    try:
        path, distance = func_timeout(args.timeout, run_algorithm, args=(args.algorithm, coords))
    except FunctionTimedOut:
        print(f"{args.algorithm.replace('_', ' ').title()} took too long to run for {dataset_filename} with {len(coords)} cities. Skipping...")
        return  # Exiting the function as the algorithm didn't complete within the timeout

    totalTime = time.time() - start_time

    if path is not None and distance is not None:
        print(f"Dataset: {dataset_filename} | Algorithm: {args.algorithm.replace('_', ' ').title()} | Path: {path} | Number of locations: {len(coords)} | Distance: {distance} | Elapsed time: {totalTime} seconds")
        
        # Remove the file extension from the dataset name
        dataset_name = os.path.splitext(dataset_filename)[0]

        # Plot the path
        plot_path(coords, path, args.algorithm, distance, totalTime, dataset_name)


# Entry point of the script
if __name__ == "__main__":
    main()
