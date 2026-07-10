# Script to run the stepwise deformation simulation
import numpy as np
import pandas as pd
import re
import sys
from collections import Counter
import random
# from MDAnalysis.lib.pkdtree import PeriodicKDTree
import extract_local_str as els
import my_common as mc
import os
from scipy.spatial import KDTree, cKDTree

# Reads the post-deformation npt relaxation log
def read_log_lammps(logfile):
    with open(logfile, 'r') as f:
        L = f.readlines()

    for i in range(len(L)):
        if 'Step' in L[i]:
            l1 = i
        if 'Loop time' in L[i]:
            l2 = i

    header = L[l1].split()
    data = []

    for line in L[l1 + 1:l2]:
        parts = line.split()
        if len(parts) == len(header):  # Skip lines that don't match the expected column count
            data.append(parts)

    data = pd.DataFrame(data, dtype='float64', columns=header)
    return data

# Reads the deformation log
def read_log_lammps_first(logfile):
    with open(logfile, 'r') as f:
        lines = f.readlines()

    start_index = None
    end_index = None

    for i, line in enumerate(lines):
        if 'Step' in line:
            start_index = i
            break

    for i, line in enumerate(lines[start_index:], start=start_index):
        if 'Loop time' in line:
            end_index = i
            break

    if start_index is None or end_index is None:
        raise ValueError("Could not find the expected 'Step' or 'Loop time' in the log file.")

    columns = lines[start_index].split()
    data = []

    for i in range(start_index + 1, end_index):
        parts = lines[i].split()
        if len(parts) == len(columns):  # Ignore lines that don't match column count
            data.append(parts)

    if not data:
        raise ValueError("No valid data found between 'Step' and 'Loop time'.")

    data = np.array(data, dtype='float64')
    df = pd.DataFrame(data, columns=columns)

    return df

def deform_relax(final_itime, step_size, path, run_lammps='lmp_serial'):
    # Run the deformation for once with the first crosslinked relaxed file. Save in tmp.dat.
    os.system('{} -in in.deform'.format(run_lammps)) # in.deform = reads cool.dat

    # Read initial log file and extract relevant info from it and save to csv
    lf = read_log_lammps('log.lammps')
    lf2 = read_log_lammps_first('log.lammps')
    Lx = lf['Lx'].iloc[-1]
    lf.to_csv(f'{path}/log_iter_0.csv')
    lf2.to_csv(f'{path}/log_noopt_0.csv')

    os.system('cp tmp.dat {}/deform_0.dat'.format(path))

    # Initialize the number of iterations
    itime = 1

    # For number of iterations we want run AND a fixed level of Lx (fixed strain value)
    while itime < final_itime and Lx < 300:
        # Update the reset_timestep xxx
        os.system("sed 's/xxx/{}/g' in.deform_opt > in.deform_opt1".format(itime*step_size))
        # Run the deform for 100 steps. Update tmp.dat.
        os.system('{} -in in.deform_opt1'.format(run_lammps))

        if itime % 10 == 0:
            os.system('cp tmp.dat {}/deform_{}.dat'.format(path, itime))

        # Read the logfile and extract the .csv and save it
        lf = read_log_lammps('log.lammps')
        lf2 = read_log_lammps_first('log.lammps')
        # Save Lx for the while condition
        Lx = lf['Lx'].iloc[-1]
        lf.to_csv('{}/log_iter_{}.csv'.format(path, itime))
        lf2.to_csv('{}/log_noopt_{}.csv'.format(path, itime))

        # Update itime
        itime += 1

import os
import glob

def deform_relax_cont(final_itime, step_size, path, run_lammps='lmp_serial'):
    # Check for existing log files to determine where to restart
    existing_logs = glob.glob(f'{path}/log_iter_*.csv')

    if existing_logs:
        # Extract iteration numbers from existing files
        iter_numbers = []
        for log_file in existing_logs:
            # Extract number from filename like 'log_iter_28.csv'
            iter_num = int(log_file.split('_')[-1].split('.')[0])
            iter_numbers.append(iter_num)

        # Start from the next iteration after the highest existing one
        start_itime = max(iter_numbers) + 1
        print(f"Resuming from iteration {start_itime} (found existing logs up to iteration {max(iter_numbers)})")

        # Read the last Lx value from the most recent log file
        import pandas as pd
        last_log_file = f'{path}/log_iter_{max(iter_numbers)}.csv'
        last_df = pd.read_csv(last_log_file)
        Lx = last_df['Lx'].iloc[-1]
        print(f"Last Lx value: {Lx}")

    else:
        # No existing logs found, start from the beginning
        start_itime = 1
        print("No existing logs found, starting from iteration 0")

        # Run the initial deformation
        os.system('{} -in in.deform'.format(run_lammps))

        # Read initial log file and extract relevant info
        lf = read_log_lammps('log.lammps')
        lf2 = read_log_lammps_first('log.lammps')
        Lx = lf['Lx'].iloc[-1]
        lf.to_csv(f'{path}/log_iter_0.csv')
        lf2.to_csv(f'{path}/log_noopt_0.csv')

    # Initialize the iteration counter
    itime = start_itime
    print(f'Lx: {Lx}')
    # Main iteration loop
    while itime < final_itime and Lx < 300:
        print(f"Running iteration {itime}...")

        # Update the reset_timestep xxx
        os.system("sed 's/xxx/{}/g' in.deform_opt > in.deform_opt1".format(itime*step_size))

        # Run the deform for the specified steps
        os.system('{} -in in.deform_opt1'.format(run_lammps))

        # Read the logfile and extract the .csv and save it
        lf = read_log_lammps('log.lammps')
        lf2 = read_log_lammps_first('log.lammps')

        # Save Lx for the while condition
        Lx = lf['Lx'].iloc[-1]

        # Save the log files
        lf.to_csv('{}/log_iter_{}.csv'.format(path, itime))
        lf2.to_csv('{}/log_noopt_{}.csv'.format(path, itime))

        print(f"Iteration {itime} completed. Lx = {Lx}")

        # Update itime
        itime += 1

    print(f"Simulation completed or stopped. Final iteration: {itime-1}, Final Lx: {Lx}")

# Usage remains the same
# deform_relax(final_itime=250, step_size=500, path=path, run_lammps=run_lammps)

def extract_final_stress_avg(final_itime, step_size, path, filename):
    # Initialize an empty dataframe
    combined_df = pd.DataFrame()
    step_list = []

    # Iterate over the range and read each CSV file
    for i in range(0, final_itime):
        log_iter = pd.read_csv(f"{path}/log_iter_{i}.csv")

        # Compute mean and standard deviation for stress components
        avg_stress = log_iter.mean().to_frame().T
        std_stress = log_iter.std().to_frame().T

        # Rename columns to indicate avg and std
        avg_stress.columns = [f"{col}_avg" for col in avg_stress.columns]
        std_stress.columns = [f"{col}_std" for col in std_stress.columns]

        # Combine avg and std into one row
        combined_row = pd.concat([avg_stress, std_stress], axis=1)

        # Append to combined dataframe
        if combined_df.empty:
            combined_df = combined_row
        else:
            combined_df = pd.concat([combined_df, combined_row], ignore_index=True)

        step_list.append(i * step_size)

    # Reset index for cleanliness
    combined_df.reset_index(drop=True, inplace=True)

    # Add step column
    combined_df['Step'] = step_list

    # Save to CSV
    combined_df.to_csv(f'{filename}.csv', index=False)

def extract_final_stress_avg2(final_itime, step_size, path, filename):
    # Initialize empty dataframes
    combined_df_all = pd.DataFrame()
    combined_df_last50 = pd.DataFrame()
    step_list = []

    # Iterate over the range and read each CSV file
    for i in range(0, final_itime):
        log_iter = pd.read_csv(f"{path}/log_iter_{i}.csv")

        # Calculate the index for last 50% of data
        last50_start_idx = len(log_iter) // 2
        log_iter_last50 = log_iter.iloc[last50_start_idx:]

        # Compute mean and standard deviation for ALL datapoints
        avg_stress_all = log_iter.mean().to_frame().T
        std_stress_all = log_iter.std().to_frame().T

        # Compute mean and standard deviation for LAST 50% datapoints
        avg_stress_last50 = log_iter_last50.mean().to_frame().T
        std_stress_last50 = log_iter_last50.std().to_frame().T

        # Rename columns to indicate avg and std for ALL data
        avg_stress_all.columns = [f"{col}_avg" for col in avg_stress_all.columns]
        std_stress_all.columns = [f"{col}_std" for col in std_stress_all.columns]

        # Rename columns to indicate avg and std for LAST 50% data
        avg_stress_last50.columns = [f"{col}_avg" for col in avg_stress_last50.columns]
        std_stress_last50.columns = [f"{col}_std" for col in std_stress_last50.columns]

        # Combine avg and std into one row for each dataset
        combined_row_all = pd.concat([avg_stress_all, std_stress_all], axis=1)
        combined_row_last50 = pd.concat([avg_stress_last50, std_stress_last50], axis=1)

        # Append to combined dataframes
        if combined_df_all.empty:
            combined_df_all = combined_row_all
            combined_df_last50 = combined_row_last50
        else:
            combined_df_all = pd.concat([combined_df_all, combined_row_all], ignore_index=True)
            combined_df_last50 = pd.concat([combined_df_last50, combined_row_last50], ignore_index=True)

        step_list.append(i * step_size)

    # Reset index for cleanliness
    combined_df_all.reset_index(drop=True, inplace=True)
    combined_df_last50.reset_index(drop=True, inplace=True)

    # Add step column to both dataframes
    combined_df_all['Step'] = step_list
    combined_df_last50['Step'] = step_list

    # Save both datasets to CSV
    combined_df_all.to_csv(f'{filename}_all_data.csv', index=False)
    combined_df_last50.to_csv(f'{filename}_last50pct.csv', index=False)

    # Print summary for user
    print(f"Saved stress analysis results:")
    print(f"  - All data: {filename}_all_data.csv")
    print(f"  - Last 50% data: {filename}_last50pct.csv")

    return combined_df_all, combined_df_last50
