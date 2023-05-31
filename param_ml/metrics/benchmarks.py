"""Script to analyse the benchmarks generated during the training of a model."""
__author__ = "David Parker"

import argparse
import traceback
import os
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

def get_cli_args():
    """Parse command line arguments"""
    try:
        parser = argparse.ArgumentParser(description="""Script to analyse the benchmarks generated during the training of a model.""")
        parser.add_argument("-i", "--input", help="Input folder containing the benchmark files.", default="data/benchmarks/")
    except:
        print("An exception occurred with argument parsing. Check your provided options.")
        traceback.print_exc()
    return parser.parse_args()


def get_gpu_benchmarks(input_folder):
    """Get the GPU benchmarks from the input folder."""
    gpu_benchmarks = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith("gpu_usage.csv"):
                gpu_benchmarks.append(os.path.join(root, file))
    gpu_benchmarks = [x for x in gpu_benchmarks if os.stat(x).st_size > 0]
    return gpu_benchmarks


def get_cpu_benchmarks(input_folder):
    """Get the CPU benchmarks from the input folder."""
    cpu_benchmarks = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith("dataset_process_time.csv"):
                cpu_benchmarks.append(os.path.join(root, file))
    return cpu_benchmarks


def make_gpu_df(gpu_benchmarks):
    """Make a dataframe of the GPU benchmarks.

    Parameters
    ----------
    gpu_benchmarks : list
        List of file paths to the GPU benchmarks.

    Returns
    -------
    pd.DataFrame
        Dataframe of the GPU benchmarks. Columns are:
        - run
        - time [s]
        - memory.used [MiB]
        - memory.free [MiB]
        - num_mols
        - num_epochs
        - batch_size
        - num_cpus
        - memory utilization [%]
    """    
    gpu_df = pd.DataFrame()
    for gpu_benchmark in gpu_benchmarks:
        # add file name as a column
        current_df = pd.read_csv(gpu_benchmark)
        current_df['run'] = pathlib.Path(gpu_benchmark).name

        # Make a time column: each row is 2 seconds apart
        current_df['time [s]'] = current_df.index * 2

        gpu_df = pd.concat([gpu_df, current_df], ignore_index=True)
    
    # Add columns based on the file name: 1000_mols_10_epochs_1_batch_1_cpus_gpu_usage.csv
    gpu_df['num_mols'] = gpu_df['run'].str.split('_').str[0]
    gpu_df['num_epochs'] = gpu_df['run'].str.split('_').str[2]
    gpu_df['batch_size'] = gpu_df['run'].str.split('_').str[4]
    gpu_df['num_cpus'] = gpu_df['run'].str.split('_').str[6]

    # Remove all text from the ' memory.used [MiB]' and ' memory.free [MiB]' columns
    gpu_df['memory.used [MiB]'] = gpu_df[' memory.used [MiB]'].str.replace(' MiB', '').astype(float)
    gpu_df['memory.free [MiB]'] = gpu_df[' memory.free [MiB]'].str.replace(' MiB', '').astype(float)
    # Drop the original columns
    gpu_df = gpu_df.drop(columns=[' memory.used [MiB]', ' memory.free [MiB]'])

    # make a utilization column
    gpu_df['memory utilization [%]'] = gpu_df['memory.used [MiB]'] / (gpu_df['memory.used [MiB]'] + gpu_df['memory.free [MiB]']) * 100


    return gpu_df


def make_cpu_df(cpu_benchmarks):
    """Make a dataframe of the CPU benchmarks.

    Parameters
    ----------
    cpu_benchmarks : list
        List of file paths to the CPU benchmarks.

    Returns
    -------
    pd.DataFrame
        Dataframe of the CPU benchmarks. Columns are:
        - dataset_name
        - process_time
        - n_mols
        - n_cpus
    """    
    cpu_df = pd.DataFrame()
    for cpu_benchmark in cpu_benchmarks:
        cpu_df = pd.concat([cpu_df, pd.read_csv(cpu_benchmark)], ignore_index=True)
    
    # Remove all whitespace from the column names
    cpu_df.columns = cpu_df.columns.str.replace(' ', '')
    
    return cpu_df


def get_gpu_stats(gpu_df):
    df = (
        gpu_df
        .groupby(['num_mols', 'batch_size'])['memory utilization [%]']
        .describe()
    )
    
    return df


def get_cpu_stats(cpu_df):
    df = (
        cpu_df
        .groupby(['n_mols', 'n_cpus'])['process_time']
        .describe()
    )
       
    return df


def main():
    args = get_cli_args()

    gpu_benchmarks = get_gpu_benchmarks(args.input)
    cpu_benchmarks = get_cpu_benchmarks(args.input)

    gpu_df = make_gpu_df(gpu_benchmarks)
    cpu_df = make_cpu_df(cpu_benchmarks)

    # Print stats
    print("GPU stats:")
    print(get_gpu_stats(gpu_df))

    print("CPU stats:")
    print(get_cpu_stats(cpu_df))

    # Plot the results

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    cpu_df.boxplot(column='process_time', by=['n_mols', 'n_cpus'], ax=axs[0])
    n_runs = cpu_df['dataset_name'].nunique()
    axs[0].set_title(f'CPU Process Times for {n_runs} runs')
    axs[0].set_ylabel('process time [s]')

    gpu_df.boxplot(column='memory utilization [%]', by=['num_mols', 'batch_size'], ax=axs[1])
    n_runs = gpu_df['run'].nunique()
    axs[1].set_title(f'GPU Memory Utilization for {n_runs} runs')
    axs[1].set_ylabel('memory utilization [%]')

    # Set the figure title
    fig.suptitle('CPU and GPU Benchmark Results')
    fig.tight_layout()

    plt.show()



if __name__ == '__main__':
    main()