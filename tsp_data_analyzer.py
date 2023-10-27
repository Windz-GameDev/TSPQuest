import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def visualize_data(df):
    # Use Seaborn's default style for the visualizations
    sns.set_theme()

    # 1. Bar Chart
    # Create a figure for the bar chart with a width of 10 inches, and height of 6 inches
    plt.figure(figsize=(10,6))
    # Create a bar plot where the x-values are distinct algorithms y-values are the averages of the totol distances by those algorithms
    sns.barplot(x='Algorithm', y='Total Distance', data=df)
    # Set title of the plot 
    plt.title('Average Total Distance by Algorithm')
    # Display plot
    plt.show()

    # 2. Line graph
    # Create a figure for the line graph
    plt.figure(figsize=(10, 6))    
    # Create a line plot to show the trend of total distance per algorithm as number of cities increase
    # Different algorithms are represented with different colors (hues)
    sns.lineplot(x='Number of Cities', y='Total Distance', hue='Algorithm', data=df, marker='o')
    # Set the title of the plot
    plt.title('Total Distance Trend by Number of Cities')
    # Display the plot
    plt.show()

    
    # 3. Scatter Plot (Needs to be tested on actual data to verify it's working)
    # Create a figure for the scatter plot 
    plt.figure(figsize=(10, 6))
    # Create a scatter plot to show the relationship between number of cities and elapsed time of an algorithm execution
    # Different algorithms are represented by different colors (hues)
    sns.scatterplot(x='Number of Cities', y='Elapsed Time (Seconds)', hue='Algorithm', data=df)
    # Set the title of the plot
    plt.title('Relation between Number of Cities and Elapsed Time')
    # Display the plot
    plt.show()

    # 4. Box Plots
    # Get the unique dataset names
    datasets = df['Dataset'].unique()

    # Iterating through each dataset and creating a box plot
    for dataset in datasets:

        # Filter dataframe to only inlcude rows whose dataset column corresponds to the current dataset to be visualized
        df_filtered = df[df['Dataset'] == dataset].dropna(subset=['Total Distance', 'Elapsed Time (Seconds)']) # Remove algorithms with NaNs to avoid breaking boxplot
    

        # Check if there are any rows left to plot
        if df_filtered.empty:
            print(f"\nNo algorithms managed to solve the dataset {dataset} in the given timeout duration or no valid data...")
            continue # Don't visualize this dataset, skip to next one


        # Create a figure for the box plot
        plt.figure(figsize=(10, 6))

        # Create a box plot to show distribution of total distances for each algorithm on a dataset
        sns.boxplot(x='Algorithm', y='Total Distance', data=df_filtered)
        # Set title of the plot
        plt.title(f'Distribution of Total Distance by Algorithm for {dataset}')
        # Display the plot
        plt.show()


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Read and perform analysis on Traveling Salesmen Algorithm performance results stored in a CSV")

    # Adding an argument for the filename
    parser.add_argument('file_path', type=str, help='Path to your TSP algorithm results CSV to be read.')

    # Parsing the arguments
    args = parser.parse_args()

    # Read and display CSV file
    df = read_csv_file(args.file_path)

    visualize_data(df)

# Entry point of the script
if __name__ == '__main__':
    main()