'''
    Source Code for Algorithms Project for COT6405

    Algorithms for Brute Force, Nearest Neighor, Branch and Bound, Dynamic programming are 
    referenced from GeeksForGeeks to ensure accurate implementation. Datasets 

    References: 
    BruteForce - https://www.geeksforgeeks.org/traveling-salesman-problem-tsp-implementation/
    Branch and Bound - https://www.geeksforgeeks.org/traveling-salesman-problem-using-branch-and-bound-2/
    Dynamic Programming - https://www.geeksforgeeks.org/travelling-salesman-problem-using-dynamic-programming/
    Greedy - https://www.geeksforgeeks.org/travelling-salesman-problem-greedy-approach/
    Datasets - http://www.math.uwaterloo.ca/tsp/world/countries.html
'''

# Import required libraries
import argparse
import math
import time 
import matplotlib.pyplot as plt
import os
from itertools import permutations
from func_timeout import func_timeout, FunctionTimedOut

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

# Branch and bound method for solving TSP
def branch_and_bound_tsp(coords):
    def calculate_lower_bound(path, remaining):
        lb = sum(euclidean_distance(coords[path[i]], coords[path[i+1]]) for i in range(len(path)-1))
        
        # Adding the minimum distance to a city not yet visited
        if remaining:
            lb += min(euclidean_distance(coords[path[-1]], coords[k]) for k in remaining)
            
            for city in remaining: 
                # Check if there are cities remaining to compare distance
                remaining_cities = remaining-{city}
                if remaining_cities:  # Check if there are still cities remaining
                    lb += min(euclidean_distance(coords[city], coords[k]) for k in remaining_cities)
        
        return lb

    best_tour = []
    best_distance = float('inf')

    def search(path, remaining):
        nonlocal best_tour, best_distance
        if not remaining:
            total_distance = sum(euclidean_distance(coords[path[i]], coords[path[i+1]]) for i in range(len(path)-1))
            total_distance += euclidean_distance(coords[path[-1]], coords[path[0]])
            
            if total_distance < best_distance:
                best_distance = total_distance
                best_tour = list(path)
            
            return

        for next_city in remaining:
            if calculate_lower_bound(path + [next_city], remaining - {next_city}) < best_distance:
                search(path + [next_city], remaining - {next_city})

    search([0], set(i for i in range(1, len(coords))))
    return best_tour, best_distance

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

def run_algorithm(algorithm, coords):
    if algorithm == 'brute_force':
        return brute_force_tsp(coords)
    elif algorithm == 'nearest_neighbor':
         return nearest_neighbor_tsp(coords)
    elif algorithm == 'branch_and_bound':
         return branch_and_bound_tsp(coords)
    elif algorithm == 'held_karp':
        return held_karp_tsp(coords)

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Solve TSP problem using different algorithms.')
    parser.add_argument('dataset', type=str, help='TSP dataset file path.')
    parser.add_argument('algorithm', type=str, choices=['brute_force', 'nearest_neighbor', 'branch_and_bound', 'held_karp'], help='Algorithm to use.')
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
