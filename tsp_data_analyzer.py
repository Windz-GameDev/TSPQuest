import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def read_csv_file(file_path):
    try:
        # Read the CSV file to dataframe, no need to explicitly set encoding as it is automatically detected by Pandas
        df = pd.read_csv(file_path)

        # Display the DataFrame
        print(df)

        return df;
    
    except FileNotFoundError:
        print(f"Error: '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred while reading '{file_path}': {e}")

def visualize_data(df, csv_basename):
    # Define the directory for saving plots
    main_plots_dir = 'tsp_analysis_charts'

    # Create the directory if it doesn't exist
    if not os.path.exists(main_plots_dir):
        os.makedirs(main_plots_dir)

    # Create the directory to store the graphics of the current that was just read
    specific_plots_dir = os.path.join(main_plots_dir, csv_basename)

    if not os.path.exists(specific_plots_dir):
        os.makedirs(specific_plots_dir)

    # 1. Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Number of Cities', y='Elapsed Time (Seconds)', hue='Algorithm', data=df)
    plt.title('Relation between Number of Cities and Elapsed Time')
    plt.savefig(os.path.join(specific_plots_dir, 'cities_vs_elapsed_time.png'))
    plt.close()

    # 2. Box Plots
    for dataset in df['Dataset'].unique():
        dataset_name, _ = os.path.splitext(dataset)  # This will remove the .tsp extension
        dataset_dir = os.path.join(specific_plots_dir, dataset_name)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        df_filtered = df[df['Dataset'] == dataset].dropna(subset=['Total Distance', 'Elapsed Time (Seconds)'])
        if df_filtered.empty:
            print(f"\nNo algorithms managed to solve the dataset {dataset} in the given timeout duration or no valid data...")
            continue

        # Bar Plot for Average Distance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Algorithm', y='Total Distance', data=df_filtered)
        plt.title(f'Average Total Distance for {dataset}')
        plt.savefig(os.path.join(dataset_dir, f'avg_distance_{dataset_name}.png'))
        plt.close()

        # Bar Plot for Average Time
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Algorithm', y='Elapsed Time (Seconds)', data=df_filtered)
        plt.title(f'Average Elapsed Time for {dataset}')
        plt.savefig(os.path.join(dataset_dir, f'avg_time_{dataset_name}.png'))
        plt.close()

        # Box Plot for Total Distance
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Algorithm', y='Total Distance', data=df_filtered)
        plt.title(f'Distribution of Total Distance by Algorithm for {dataset}')
        plt.savefig(os.path.join(dataset_dir, f'distribution_total_distance_{dataset_name}.png'))
        plt.close()

        # Box Plot for Elapsed Time
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Algorithm', y='Elapsed Time (Seconds)', data=df_filtered)
        plt.title(f'Distribution of Elapsed Time by Algorithm for {dataset}')
        plt.savefig(os.path.join(dataset_dir, f'distribution_elapsed_time_{dataset_name}.png'))
        plt.close()

    print(f"\nSaved charts to the {specific_plots_dir} directory\n")

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Read and perform analysis on Traveling Salesmen Algorithm performance results stored in a CSV")

    # Adding an argument for the filename
    parser.add_argument('file_path', type=str, help='Path to your TSP algorithm results CSV to be read.')

    # Parsing the arguments
    args = parser.parse_args()

    print("\n")

    # Read and display CSV file
    df = read_csv_file(args.file_path)

    # Proceed only if the dataframe was successfully created
    if df is not None:
        csv_basename = os.path.splitext(os.path.basename(args.file_path))[0]
        visualize_data(df, csv_basename)
    else:
        print("Unable to proceed due to an error with the CSV file.")

# Entry point of the script
if __name__ == '__main__':
    main()