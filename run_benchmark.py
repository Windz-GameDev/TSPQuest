import subprocess
import argparse
import csv
import os

# List of algorithms to run, be careful about changing this, adding unsupported algorithms may lead to unexpected results 
algorithms = ['brute_force', 'nearest_neighbor', 'branch_and_bound', 'held_karp', 'two_opt', 'christofides', 'ant_colony', 'simulated_annealing']

def main():
    # Setting up argument parsing
    parser = argparse.ArgumentParser(description="Run the benchmark script for all algorithms with the specified dataset and write results to a CSV file.")
    parser.add_argument('dataset', help='Path to the dataset file')
    parser.add_argument('csv_file', help='Name of the CSV file to save results. Do not add file extension, it is handled automatically.')
    parser.add_argument('--timeout', type=int, default=60, help='Timeout for each algorithm in seconds. Default is 60')
    args = parser.parse_args()

    # Directory for benchmark results
    results_dir = 'benchmark_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # CSV filename with timeout
    csv_filename = f"{args.csv_file}_timeout_{args.timeout}.csv"
    csv_file_path = os.path.join(results_dir, csv_filename)

    file_exists = os.path.isfile(csv_file_path) and os.path.getsize(csv_file_path) > 0
    dataset_filename = os.path.basename(args.dataset)

    # Write the header to a CSV file
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # If the file didn't exist or was empty, write the header
        if not file_exists:
            writer.writerow(['Dataset', 'Algorithm', 'Number of Cities', 'Total Distance', 'Elapsed Time (Seconds)'])

        # Run the benchmark.py script with different algorithms and capture the output
        for algorithm in algorithms:
            command = ['python', 'benchmark.py', args.dataset, algorithm, '--timeout', str(args.timeout)]
            completed_process = subprocess.run(command, capture_output=True, text=True)

            algorithm_name = algorithm.replace('_', ' ').title()
            
            # Printing the return code
            print(f"\nAlgorithm: {algorithm_name}, Return Code: {completed_process.returncode}")

            # Get the output and print it
            output = completed_process.stdout.strip()
            print(f"Output:\n{output}")

            if "took too long to run" in output:
                parts = output.split(' ')
                num_cities = parts[-3] # Extract number of cities from the output, may need to change index if skipping output is changed
                writer.writerow([dataset_filename, algorithm_name, num_cities, 'NaN', 'NaN'])
                continue; # Skip the remaining part of the loop

            # Parse the output to extract key information
            parts = output.split(' | ')
            num_cities = parts[3].split(': ')[1]
            total_distance = parts[4].split(': ')[1]
            elapsed_time = parts[5].split(': ')[1].split(' ')[0] # Get time value without seconds

            # Write information to the CSV file
            writer.writerow([dataset_filename, algorithm_name, num_cities, total_distance, elapsed_time])

if __name__ == '__main__':
    main()
