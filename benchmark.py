# Import required libraries
import argparse
import math
from itertools import permutations

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def read_tsp_file_from_disk(filename):
    with open(filename, 'r') as f:
        return read_tsp_file(f.read())

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

# Brute-force algorithm for solving TSP
def brute_force_tsp(coords):
    min_distance = float('inf')  # Initialize min_distance to infinity
    best_path = None  # Initialize best_path to None
    
    # Generate all possible permutations and calculate distance
    for path in permutations(range(len(coords))):
        distance = 0
        for i in range(len(path) - 1):
            distance += euclidean_distance(coords[path[i]], coords[path[i+1]])
        distance += euclidean_distance(coords[path[-1]], coords[path[0]])
        
        # Update min_distance and best_path
        if distance < min_distance:
            min_distance = distance
            best_path = path
    
    return best_path, min_distance

# Greedy algorithm for solving TSP
def greedy_tsp(coords):
    num_cities = len(coords)
    unvisited = set(range(num_cities))
    path = []
    current_city = 0  # Start from the first city
    path.append(current_city)
    unvisited.remove(current_city)
    total_distance = 0
    
    # Visit nearest unvisited city
    while unvisited:
        nearest_city = min(unvisited, key=lambda city: euclidean_distance(coords[current_city], coords[city]))
        total_distance += euclidean_distance(coords[current_city], coords[nearest_city])
        current_city = nearest_city
        path.append(current_city)
        unvisited.remove(current_city)
    
    total_distance += euclidean_distance(coords[current_city], coords[path[0]])
    return path, total_distance

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Solve TSP problem using different algorithms.')
    parser.add_argument('dataset', type=str, help='TSP dataset file path.')
    parser.add_argument('algorithm', type=str, choices=['brute_force', 'greedy'], help='Algorithm to use.')
    args = parser.parse_args()
    
    # Read dataset and execute algorithm
    coords = read_tsp_file_from_disk(args.dataset)  # <-- Modified this line
    if args.algorithm == 'brute_force':
        path, distance = brute_force_tsp(coords)
        print(f"Brute-force: Path = {path}, Distance = {distance}")
    elif args.algorithm == 'greedy':
        path, distance = greedy_tsp(coords)
        print(f"Greedy: Path = {path}, Distance = {distance}")


# Entry point of the script
if __name__ == "__main__":
    main()
