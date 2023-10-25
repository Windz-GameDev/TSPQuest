'''
This script generates a Traveling Salesman Problem (TSP) dataset with specific
parameters. It allows customization for the number of locations, the dataset name,
comments, and coordinate boundaries.

The generated dataset is structured based on the datasets from:
https://www.math.uwaterloo.ca/tsp/world/countries.html

The program is intended for use with the benchmark.py file, which is contained in
the same repository.
'''

import argparse
import random

# Function to generate a TSP dataset with specified parameters
def generate_tsp_dataset(num_locations, name="Random", comment="Randomly generated TSP dataset", min_coord=10000, max_coord=30000):
    dataset = []
    # Add dataset information to file which may be used during file processing
    dataset.append(f"NAME : {name}")
    dataset.append(f"COMMENT : {num_locations} locations in {comment}")
    dataset.append(f"TYPE : TSP")
    dataset.append(f"DIMENSION : {num_locations}")
    dataset.append("EDGE_WEIGHT_TYPE : EUC_2D")
    dataset.append("NODE_COORD_SECTION")

    # Generate random coordinates for each location, starting at index 1 to ensure coordinates are correctly labeled in the file
    for i in range(1, num_locations + 1):
        x = random.uniform(min_coord, max_coord)
        y = random.uniform(min_coord, max_coord)
        dataset.append(f"{i} {x:.4f} {y:.4f}") # Coordinates are rounded to four decimals to match the official datasets

    dataset.append("EOF") # Mark the end of file so benchmark.py knows when to stop processing

    return "\n".join(dataset) # Combine each string in the dataset list into a single string, separating each with a newline character

def write_to_file(filename, data):
    with open(filename, "w") as file:
        file.write(data)

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Generate a TSP dataset.")
    parser.add_argument('num_locations', type=int, help='Number of locations in the dataset.')
    parser.add_argument('filename', type=str, help='Filename to save the dataset')
    parser.add_argument('--name', type=str, default='Random', help='Name of the dataset')
    parser.add_argument('--comment', type=str, default='Randomly generated TSP dataset.', help='A commment to include in the dataset.')
    parser.add_argument('--min_coord', type=int, default = 10000, help='Minimum coordinate value.')
    parser.add_argument('--max_coord', type=int, default=30000, help='Maximum coordinate value.')

    args = parser.parse_args() # Parse arguments

    # Generate the TSP dataset using the passed arguments and write it to a file
    tsp_data = generate_tsp_dataset(args.num_locations, args.name, args.comment, args.min_coord, args.max_coord) 
    write_to_file(f"Datasets/{args.filename}", tsp_data)

# Entry point of the script
if __name__ == "__main__":
    main()
