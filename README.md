# TSPQuest

This repository contains a benchmark Python script for solving the Traveling Salesman Problem (TSP) using different algorithms and analyzing those results.

## Table of Contents

- [Description](#description)
- [Algorithms](#algorithms)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Running the TSP Generator Script](#running-the-tsp-generator-script)
  - [Running the Benchmark Script](#running-the-benchmark-script)
- [Dependencies](#dependencies)
- [References](#references)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)
- [Future Work](#future-work)

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

## Usage

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

## Running the Benchmark Script

Execute the benchmark script by using the following command structure in your terminal:

```bash
python benchmark.py Datasets/<dataset> <algorithm> [options]
```

### Parameters

- `<dataset>`: (Required) The name of the TSP dataset file you wish to use (e.g., `lu98.tsp`).
- `<algorithm>`: (Required) Specify the algorithm you want to use. Available algorithms include:
  - `brute_force`
  - `branch_and_bound`
  - `held_karp`
  - `nearest_neighbor`
  - `two_opt`
  - `simulated_annealing`

### Options

- `--timeout <# seconds>`: Specify a maximum number of seconds any one algorithm may attempt to solve the TSP. The default is 60 seconds.

### Examples

```bash
python benchmark.py Datasets/lu98.tsp brute_force
```

This example command will run the brute-force algorithm on the `lu98.tsp` dataset.

```bash
python benchmark.py Datasets/lu98.tsp brute_force --timeout 60
```

This command will run the brute-force algorithm on the `lu98.tsp` dataset with a timeout of 60 seconds.

## Running the `tsp_data_analyzer.py` Script

Execute the TSP data analyzer script by using the following command structure in your terminal:

```bash 
python tsp_data_analyzer.py <input_file>
```

### Parameters

- `<input_file>`: (Required) Specify the path to the CSV file containing the TSP algorithm performance results.

### Example

```bash
python tsp_data_analyzer.py data.csv
```

This example command will:
- Read the CSV file named `data.csv`.
- Perform the analysis on the TSP algorithm results contained in the file.
- Display the resulting visualizations.

Please replace `data.csv` with the path to your actual CSV file. The CSV file should be in the correct format with the required columns ('Algorithm', 'Total Distance', 'Number of Cities', 'Elapsed Time (Seconds)', 'Dataset'). See the example CSV in this repository for reference.

## Dependencies

The scripts require Python and the following Python libraries:

- `argparse`
- `math`
- `itertools`
- `matplotlib`
- `seaborn`
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
