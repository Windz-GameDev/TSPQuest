# TSPQuest

This repository contains a benchmark Python script for solving the Traveling Salesman Problem (TSP) using different algorithms and analyzing those results.

## Description

The Traveling Salesman Problem (TSP) is a classic optimization problem in computer science. The goal is to find the shortest possible route that visits a set of cities and returns to the origin city. This repository provides implementations of various algorithms to solve the TSP, ranging from exact methods to heuristic and approximation algorithms.
This repository was created for our COT6405 class project - "The Traveling Salesman, Comparison of Efficiency between exact and approximate heuristic algorithms".

## Algorithms

Note: This Repo is a work in progress; not all algorithms have been implemented yet.

The script in this repository implements the following algorithms for solving the TSP:

1. **Brute Force**: An exact algorithm that generates all possible tours and then selects the shortest one. Time Complexity: \(O(n!)\).

2. **Branch and Bound**: An exact algorithm that uses a tree search strategy. Time Complexity: Varies, best case \(O(n^2 \log n)\), worst case \(O(2^n)\).

3. **Held-Karp Algorithm**: An exact algorithm that uses dynamic programming. Time Complexity: \(O(n^2 2^n)\).

4. **Nearest Neighbor (NN)**: A heuristic algorithm. Time Complexity: \(O(n^2)\).

5. **2-Opt Algorithm**: A heuristic algorithm. Time Complexity: \(O(n^2)\) for each 2-opt move, \(O(n^3)\) if applied exhaustively.

6. **Simulated Annealing**: A probabilistic technique. Time Complexity: Varies depending on parameters, typically \(O(n^2)\) per iteration for TSP.

## Dataset

The script uses the following TSP datasets, which were retrieved from [The University of Waterloo's TSP page](https://www.math.uwaterloo.ca/tsp/world/countries.html):

- `lu98.tsp`: Represents the locations of 980 cities in Luxembourg.
- `qa194.tsp`: Represents 194 locations in Qatar.
- `wi4.tsp`: Represents the locations of 4 cities in Western Sahara. (A trimmed down version of the next dataset which we modified.)
- `wi29.tsp`: Represents the locations of 29 cities in Western Sahara.
- `ja9847.tsp`: Represents the locations of 9,847 cities in Japan.

We also use our own TSP dataset generator, which is based on the file structure of the previously listed datasets.

It can be run with the following command:

## Running the TSP Generator Script

Execute the TSP generator script by using the following command structure in your terminal:

```bash
python tsp_generator.py <num_cities> <output_file> [options]
```

### Parameters

- `<num_cities>`: (Required) The number of cities you want to generate in the TSP dataset.
- `<output_file>`: (Required) Specify the name of the file where the generated TSP dataset will be saved.

### Options

- `--name <dataset_name>`: Specify the name you wish to assign to your dataset.
- `--comment "<comment>"`: Add a comment or description that will be included in the dataset.
- `--min_coord <min_coordinate>`: Specify the minimum coordinate value for the cities in the dataset.
- `--max_coord <max_coordinate>`: Specify the maximum coordinate value for the cities in the dataset.

### Example

```bash
python tsp_generator.py 10 output.tsp --name MyDataset --comment "My custom comment" --min_coord 5000 --max_coord 25000
```

This example command will:
- Generate a TSP dataset with 10 cities.
- Save it to a file named `output.tsp`.
- Name the dataset "MyDataset".
- Include a custom comment "My custom comment".
- Cities will have coordinates ranging between 5000 and 25000.


## Usage

To run the script, use the following command:

python benchmark.py Datasets/`dataset` `algorithm` `--timeout [# seconds]`

Replace `dataset` with the name of the TSP dataset file (e.g., `lu98.tsp`), and `algorithm` with the algorithm you want to use (`brute_force`, `branch_and_bound`, `held_karp`, `nearest_neighbor`, `two_opt`, or `simulated_annealing`).

For example, to solve the `lu98.tsp` dataset using the brute-force algorithm, run:

python benchmark.py Datasets/lu98.tsp brute_force

By specifying a timeout, you specify a maximum number of seconds any one algorithm may attempt to solve the TSP, the default is 60.

python benchmark.py Datasets/lu98.tsp brute_force --timeout 60

## Dependencies

The scripts require Python and the following Python libraries:

- `argparse`
- `math`
- `itertools`
- `matplotlib`
- `func_timeout`

## References

GeeksForGeeks code implementations for various algorithms were heavily referenced in this study: https://www.geeksforgeeks.org/

- Brute Force
- Branch and Bound
- Dynammic Programming
- Nearest Neighbor

## Authors

Aaron Goldstein, Jonathan O'Berry

## Acknowledgements

We would like to thank our professors and peers for their valuable feedback and support throughout this project. We would also like to acknowledge the creators of the datasets we used for our analysis, and GeeksForGeeks for their explanations of the various algorithms and example implementations.

## Future Work

Future work could involve testing these algorithms on larger datasets beyond 10000 cities to assess scalability. Exploring metaheuristic algorithms like Genetic Algorithms or Ant Colony Optimization could also be beneficial. Machine learning techniques, particularly reinforcement learning, could offer a novel approach to solving the TSP. Additionally, it would be interesting to investigate the impact of problem constraints such as time-windows or vehicle capacities on the performance of these algorithms.
