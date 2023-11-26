'''
This script generates a Traveling Salesman Problem (TSP) dataset with specific
parameters. It allows customization for the number of locations, the dataset name,
comments, and coordinate boundaries. It aditionally allows for generating datasets
which form a metric space, satisfying the requirements for the christofides algorithm.
It should be noted however that generating a dataset forming a metric space 
can be especially time consuming and this should not be done for large datasets.

The generated dataset is structured based on the datasets from:
https://www.math.uwaterloo.ca/tsp/world/countries.html

The program is intended for use with the benchmark.py file, which is contained in
the same repository.
'''

import argparse
import random

# Function to generate a TSP dataset with specified parameters
def generate_tsp_dataset(num_locations, name="Random", comment="Randomly generated TSP dataset", min_coord=10000, max_coord=30000, metric_space=False):
    dataset = []
    # Add dataset information to file which may be used during file processing
    dataset.append(f"NAME : {name}")
    dataset.append(f"COMMENT : {num_locations} locations in {comment}")
    dataset.append(f"TYPE : TSP")
    dataset.append(f"DIMENSION : {num_locations}")
    dataset.append("EDGE_WEIGHT_TYPE : EUC_2D")
    dataset.append("NODE_COORD_SECTION")

    while True:
        # Generate random coordinates for each location
        coords = [(random.uniform(min_coord, max_coord), random.uniform(min_coord, max_coord)) for _ in range(num_locations)]

        if metric_space:
            # Check that the triangle inequality holds for all triples of points
            valid = True
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    for k in range(j + 1, len(coords)):
                        dist_ij = ((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2) ** 0.5
                        dist_jk = ((coords[j][0] - coords[k][0]) ** 2 + (coords[j][1] - coords[k][1]) ** 2) ** 0.5
                        dist_ki = ((coords[k][0] - coords[i][0]) ** 2 + (coords[k][1] - coords[i][1]) ** 2) ** 0.5
                        if dist_ij > dist_jk + dist_ki or dist_jk > dist_ij + dist_ki or dist_ki > dist_ij + dist_jk:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break

            if valid:
                break
        else:
            break

    for i in range(1, num_locations + 1):
        dataset.append(f"{i} {coords[i-1][0]:.4f} {coords[i-1][1]:.4f}") # Coordinates are rounded to four decimals to match the official datasets

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
    parser.add_argument('--metric_space',  action='store_true', help='Whether the dataset should form a metric space. WARNING: Computationally expensive for large numbers of locations!')

    args = parser.parse_args() # Parse arguments

    # Generate the TSP dataset using the passed arguments and write it to a file
    tsp_data = generate_tsp_dataset(args.num_locations, args.name, args.comment, args.min_coord, args.max_coord, args.metric_space)
    write_to_file(f"Datasets/{args.filename}", tsp_data)

# Entry point of the script
if __name__ == "__main__":
    main()
