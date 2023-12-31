# TSPQuest

This repository contains a benchmark Python script for solving the Euclidean Traveling Salesman Problem (TSP) using different algorithms. It also containers various other scripts for generating problem datasets, running all the algorithms at once on a single dataset and writing those results to a CSV, and an analyzer script for generating visualizations for the data.

## Table of Contents

- [Description](#description)
- [Algorithms](#algorithms)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Running the `tsp_dataset_generator.py` Script](#running-the-tsp_dataset_generatorpy-script)
  - [Running the `benchmark.py` Script](#running-the-benchmarkpy-script)
  - [Running the `run_benchmark.py` Script](#running-the-run_benchmarkpy-script)
  - [Running the `tsp_data_analyzer.py` Script](#running-the-tsp_data_analyzerpy-script)
- [Dependencies](#dependencies)
  - [Standard Libraries](#standard-libraries-pre-installed-with-python)
  - [External Libraries](#external-libraries-need-to-be-installed-separately)
- [References](#references)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)
- [Future Work](#future-work)

## Description

The Traveling Salesman Problem (TSP) is a classic optimization problem in computer science. The goal is to find the shortest route that visits a set of cities and returns to the origin city. This repository implements various algorithms to solve the TSP, ranging from exact methods to heuristic and approximation algorithms.
This repository was created for our COT6405 class project - "The Euclidean Traveling Salesman, Comparison of Efficiency between Exact and Approximate Heuristic Algorithms."

## Algorithms

The script in this repository implements the following algorithms for solving the TSP:

1. **Brute Force**: An exact algorithm that generates all possible tours and selects the shortest one. The worst-case time complexity of our implementation is: \(O(n! * n)\).

2. **Branch and Bound**: An exact algorithm that uses a tree search strategy. The worst-case time complexity of our implementation is: \(O(n! * n^2)\); this is due to matrix operations used for estimating lower bounds. In practice, it is faster than Brute Force.

3. **Held-Karp Algorithm**: An exact algorithm that uses dynamic programming. The time Complexity of our implementation is: \(O(n^2 2^n)\).

4. **Nearest Neighbor (NN)**: A heuristic algorithm. The time complexity of our implementation is: \(O(n^2)\).

5. **2-Opt Algorithm**: A heuristic algorithm. Time Complexity: \(O(n^2)\) for each 2-opt move, \(O(n^3)\) if applied exhaustively.

6. **Christofides Algorithm**: A heuristic algorithm that combines a minimum spanning tree and a minimum weight graph. Time Complexity: \(O(n^3)\).

7. **Ant Colony Optimization Algorithm**: A metaheuristic algorithm based on the pheromone navigation of ants. Time Complexity: \(O(m n^2)\)

8. **Simulated Annealing**: A probabilistic technique. Time Complexity: Depending on parameters, typically \(O(n^2)\) per iteration for TSP.

## Dataset

The script uses the following TSP datasets, which were retrieved from [The University of Waterloo's TSP page](https://www.math.uwaterloo.ca/tsp/world/countries.html):

- `wi4.tsp`: Represents the locations of 4 cities in Western Sahara. (A trimmed-down version of the next dataset, which we modified.)
- `wi29.tsp`: Represents the locations of 29 cities in Western Sahara.
- `qa194.tsp`: Represents 194 locations in Qatar.
- `lu980.tsp`: Represents the locations of 980 cities in Luxembourg.
- `ja9847.tsp`: Represents the locations of 9,847 cities in Japan.

We also use our own TSP dataset generator, which is based on the file structure of the previously listed datasets.

## Usage

## Running the `tsp_dataset_generator.py` Script

The purpose of the tsp_dataset_generator.py script is to generate unique datasets on which one may run the various algorithms implemented in `benchmark.py`

Execute the TSP generator script by using the following command structure in your terminal:

```bash
python tsp_generator.py <num_cities> <output_file> [options]
```

### Parameters

- `<num_cities>`: (Required) The number of cities you want to generate in the TSP dataset.
- `<output_file>`: (Required) Specify the file name where the generated TSP dataset will be saved. Add the .tsp file extension for this script.

### Options

- `--name <dataset_name>`: Specify the name you wish to assign to your dataset.
- `--comment "<comment>"`: Add a comment or description that will be included in the dataset.
- `--min_coord <min_coordinate>`: Specify the minimum coordinate value for the cities in the dataset.
- `--max_coord <max_coordinate>`: Specify the maximum coordinate value for the cities in the dataset.
- `--metric_space`: Adding this tag will guarantee the TSP generator creates a problem dataset that forms a metric space, which is required for the Christofides Algorithm. We do not recommend using this tag for inputs greater than 700 or if you have a weak CPU.

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

## Running the `benchmark.py` Script

Execute the benchmark script by using the following command structure in your terminal:

```bash
python benchmark.py datasets/<dataset> <algorithm> [options]
```

### Parameters

- `<dataset>`: (Required) The name of the TSP dataset file you wish to use (e.g., `lu980.tsp`).
- `<algorithm>`: (Required) Specify the algorithm you want to use. Available algorithms include:
  - `brute_force`
  - `branch_and_bound`
  - `held_karp`
  - `nearest_neighbor`
  - `two_opt`
  - `christofides`
  - `ant_colony`
  - `simulated_annealing`
  

### Options

- `--timeout <# seconds>`: Specify a maximum number of seconds any one algorithm may attempt to solve the TSP. The default is 60 seconds.

### Examples

```bash
python benchmark.py datasets/lu980.tsp nearest_neighbor
```

This example command will:
  - Run the nearest neighbor algorithm on the `lu980.tsp` dataset.
  - It will have the default timeout of 60 seconds.
  - It will display the dataset used, algorithm used, path, number of cities, total distance, and the elapsed time.
  - It will save a scatterplot visualizing the path at `path_results/{dataset_name}` in a PNG file.
  - If it times out, it will display skipping the algorithm.

```bash
python benchmark.py datasets/lu980.tsp brute_force --timeout 5
```

This example command will:
  - Run the brute force algorithm on the `lu980.tsp` dataset.
  - It will have a timeout of 5 seconds.
  - If it finishes, it will display the dataset, the algorithm, the path, the number of cities, the total distance, and the elapsed time.
  - If finished, it will save a scatterplot visualizing the path at `path_results/{dataset_name}` in a PNG file.
  - If it times out, it will display skipping the algorithm.
  - Note: This particular command is pretty much 100% guaranteed to time out.

## Running the `run_benchmark.py` Script

Execute the `run_benchmark.py` script by using the following command structure in your terminal:

```bash
python run_benchmark.py <dataset> <csv_file> [options]
```

### Parameters
- `<dataset>`: (Required) Specify the path to the dataset file.
- `<csv_file>`: (Required) CSV file Name to save the results. Please don't add file extension; it is handled automatically. If CSV already exists, the data will be appended; a new file will be created if it doesn't.

### Options
-`--timeout <# seconds>`: Specify how long any algorithm will run at max in seconds. The default is 60 seconds. 

### Examples

```bash
python run_benchmark.py datasets/wi4.tsp results
```

This example command will:
- This command will print the results of each algorithm on the `wi4.tsp` dataset to the console.
- The results will be stored in the `results_timeout_60.csv` file in the benchmark_results directory.
- Each algorithm gets the default 60 seconds to run.
- If any algorithm times out, NaN will be filled in for distance and elapsed time for that algorithm on the CSV.
 
```bash
python run_benchmark.py datasets/lu980.tsp results --timeout 5
```

This example command will:
- This command will print the results of each algorithm on the `lu980.tsp` dataset
- The results will be stored in the `results_timeout_5.csv` file in the benchmark_results directory.
- Any single algorithm gets 5 seconds at most to run.
- If any algorithm times out, NaN will be filled in for distance and elapsed time in the CSV for that algorithm.

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
- Read the CSV file named `data.csv`
- Perform an analysis of the TSP algorithm results contained in the file.
- Save the visualizations for each dataset at the following directory `tsp_analysis_charts/data(name of CSV)/{dataset name}`
- Save the overall scatterplot visualization for the number of cities and time in the `tsp_analysis_charts/data(name of CSV)` directory.

Please replace `data.csv` with the path to your actual CSV file. The CSV file should be in the correct format with the required columns ('Dataset', 'Algorithm', 'Number of Cities', 'Total Distance', 'Elapsed Time (Seconds)'). Please look at the example CSV in the `benchmark_results` directory in this repository for reference.

## Dependencies

The scripts require Python and the following Python libraries:

### Standard Libraries (Pre-installed with Python)
- `sys`
- `argparse`
- `math`
- `time`
- `os`
- `heapq`
- `random`
- `itertools`
- `subprocess`
- `csv`
- `copy`
- `heapq`
  
### External Libraries (Need to be installed separately)
- `matplotlib`
- `networkx`
- `numpy`
- `func_timeout`
- `mpl_toolkits`
- `seaborn`
- `pandas`
  
## References

\[1\] GeeksforGeeks, “Traveling salesman problem (TSP) implementation,” GeeksforGeeks, https://www.geeksforgeeks.org/traveling-salesman-problem-tsp-implementation/. 

\[2\] A. Bari, “7.3 traveling salesman problem - branch and bound,” YouTube, https://www.youtube.com/watch?v=1FEP_sNb62k&amp;t=515s. 

\[3\] GeeksforGeeks, “Traveling salesman problem using branch and bound,” GeeksforGeeks, https://www.geeksforgeeks.org/traveling-salesman-problem-using-branch-and-bound-2/. 

\[4\] GeeksforGeeks, “Travelling salesman problem using Dynamic Programming,” GeeksforGeeks, https://www.geeksforgeeks.org/travelling-salesman-problem-using-dynamic-programming/. 

\[5\] C. Keirua, “Traveling salesman problem with 2-Opt,” KeiruaProd, https://www.keiruaprod.fr/blog/2021/09/15/traveling-salesman-with-2-opt.html. 

\[6\] GeeksforGeeks, “Prim’s algorithm for minimum spanning tree (MST),” GeeksforGeeks, https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/. 

\[7\] NetworkX, “Network X documentation for compose,” NetworkX, https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.operators.binary.compose.html. 

\[8\] M. Gazda, “Christofides algorithm in Python,” Matej Gazda, http://matejgazda.com/christofides-algorithm-in-python/. 

\[9\] A. Buchanan, “TSP christofides algorithm,” YouTube, https://www.youtube.com/watch?v=Uu2ptesvteE. 

\[10\] “Christofides algorithm,” notebook.community, https://notebook.community/DhashS/Olin-Complexity-Final-Project/code/03_approximation_algorithms. 

\[11\] NetworkX, “Converting to and from other data formats,” NetworkX 3.2.1 documentation, https://networkx.org/documentation/stable/reference/convert.html. 

\[12\] NetworkX, “Network X documentation for Christofides,” NetworkX, https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.approximation.traveling_salesman.christofides.html. 

\[13\] GeeksforGeeks, “Travelling salesman problem: Greedy Approach,” GeeksforGeeks, https://www.geeksforgeeks.org/travelling-salesman-problem-greedy-approach/. 

\[14\] I. Ramamurthy, “Implementing ant colony optimization in python- solving traveling salesman problem,” Medium, https://induraj2020.medium.com/implementation-of-ant-colony-optimization-using-python-solve-traveling-salesman-problem-9c14d3114475#:~:text=Implementing%20Ant%20colony%20optimization%20in%20python%2D%20solving%20Traveling%20salesman%20problem,-Induraj&amp;text=Ant%20colony%20optimization%20(ACO)%20is,by%20the%20behavior%20of%20ants. 

\[15\] J. Bakulikira, “Ant Colony Traveling Salesman Algorithm,” GitHub, https://github.com/Josephbakulikira/Traveling-Salesman-Algorithm/blob/master/antColony.py. 

\[16\] “Ant colony optimization using Python,” YouTube, https://www.youtube.com/watch?v=EJKdmEbGre8 (accessed Nov. 24, 2023). 

\[17\] C. W. Alvarenga, “How to implement simulated annealing algorithm in Python,” Medium, https://medium.com/swlh/how-to-implement-simulated-annealing-algorithm-in-python-ab196c2f56a0. 

\[18\] J. Brownlee, “Simulated annealing from scratch in Python,” Machine Learning Mastery, https://machinelearningmastery.com/simulated-annealing-from-scratch-in-python/. 

\[19\] M. Perry, “simanneal,” GitHub, https://github.com/perrygeo/simanneal/blob/master/simanneal/anneal.py (accessed Nov. 25, 2023). 

\[20\] N. Rooy, “Effective simulated annealing with python,” nathanrooy.github.io, https://nathanrooy.github.io/posts/2020-05-14/simulated-annealing-with-python/. 

\[21\] F. Liang, “Optimization techniques-simulated annealing,” Medium, https://towardsdatascience.com/optimization-techniques-simulated-annealing-d6a4785a1de7. 

\[22\] Baeldung contributors, “Simulated annealing explained,” Baeldung on Computer Science, https://www.baeldung.com/cs/simulated-annealing (accessed Nov. 25, 2023). 

\[23\] “Simulated annealing algorithm in python - travelling salesperson problem,” YouTube, https://www.youtube.com/watch?v=35fzyblVdmA. 

\[24\] W. Cook, “National Traveling Salesman Problems,” Solving TSPs, https://www.math.uwaterloo.ca/tsp/world/countries.html (accessed Nov. 25, 2023). 

## Authors

Aaron Goldstein, Jonathan O'Berry

## Acknowledgements

We thank our professors and peers for their valuable feedback and support throughout this project. We would also like to acknowledge the creators of the datasets we used for our analysis and our various sources for their explanations of the various algorithms and example implementations.
