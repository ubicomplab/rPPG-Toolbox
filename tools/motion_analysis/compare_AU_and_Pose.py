# AUs (Non-rigid) and Pose (Rigid) Comparison Script
# 
# This is a simple script that serves as an example to compare
# non-rigid and rigid motion analysis for two datasets from 
# OpenFace to generate a plot.
#
# See comments and the motion_analysis folder README For more details.

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Define the 17 facial action units (AUs) of interest
    aus_of_interest = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]

    # Define the 3 pose rotations of interest
    pose_rotations_of_interest = ['pose_Rx', 'pose_Ry', 'pose_Rz']

    # Define the directory where the OpenFace CSV files are located
    input_dir = 'path/to/UBFC-rPPG'
    compare_dir = '/path/to/MAUBFC-rPPG'

    # Initialize empty lists to store the mean standard deviation values
    au_intensity_std_means = []
    pose_rotation_std_means = []

    # Loop over all CSV files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            # if 'hp' not in filename:
            #     continue
            # Load the CSV file into a pandas DataFrame
            df = pd.read_csv(os.path.join(input_dir, filename))
            
            # Extract the pose rotation data from the DataFrame
            pose_rotations = df[['frame', 'timestamp'] + pose_rotations_of_interest]

            # Compute the standard deviation of each pose rotation
            pose_rotation_std = pose_rotations[[col for col in pose_rotations.columns if col not in ['frame', 'timestamp']]].std()

            # Compute the mean of the standard deviations of all pose rotations
            pose_rotation_std_mean = pose_rotation_std.mean()
            
            # Append the mean standard deviation value to the list
            pose_rotation_std_means.append(pose_rotation_std_mean)

            # Extract the AU intensity data from the DataFrame
            aus_intensity = df[['frame', 'timestamp'] + ['AU{:02d}_r'.format(au) for au in aus_of_interest]]

            # Compute the standard deviation of each AU intensity
            au_intensity_std = aus_intensity[[col for col in aus_intensity.columns if col not in ['frame', 'timestamp']]].std()

            # Compute the mean of the standard deviations of all AUs
            au_intensity_std_mean = au_intensity_std.mean()
            
            # Append the mean standard deviation value to the list
            au_intensity_std_means.append(au_intensity_std_mean)

    # Calculate Overall Means and Medians
    au_mean = np.mean(au_intensity_std_means)
    au_median = np.median(au_intensity_std_means)
    pose_mean = np.mean(pose_rotation_std_means)
    pose_median = np.median(pose_rotation_std_means)

    # Initialize empty lists to store the mean standard deviation values
    compare_au_intensity_std_means = []
    compare_pose_rotation_std_means = []

    # Loop over all CSV files in the input directory
    for filename in os.listdir(compare_dir):
        if filename.endswith('.csv'):
            # if 'hp' not in filename:
            #     continue
            # Load the CSV file into a pandas DataFrame
            df = pd.read_csv(os.path.join(compare_dir, filename))
            
            # Extract the pose rotation data from the DataFrame
            pose_rotations = df[['frame', 'timestamp'] + pose_rotations_of_interest]

            # Compute the standard deviation of each pose rotation
            pose_rotation_std = pose_rotations[[col for col in pose_rotations.columns if col not in ['frame', 'timestamp']]].std()

            # Compute the mean of the standard deviations of all pose rotations
            pose_rotation_std_mean = pose_rotation_std.mean()
            
            # Append the mean standard deviation value to the list
            compare_pose_rotation_std_means.append(pose_rotation_std_mean)

            # Extract the AU intensity data from the DataFrame
            aus_intensity = df[['frame', 'timestamp'] + ['AU{:02d}_r'.format(au) for au in aus_of_interest]]

            # Compute the standard deviation of each AU intensity
            au_intensity_std = aus_intensity[[col for col in aus_intensity.columns if col not in ['frame', 'timestamp']]].std()

            # Compute the mean of the standard deviations of all AUs
            au_intensity_std_mean = au_intensity_std.mean()
            
            # Append the mean standard deviation value to the list
            # compare_au_intensity_std_means.append(au_intensity_std_mean/pose_rotation_std_mean)
            compare_au_intensity_std_means.append(au_intensity_std_mean)

    # Plot comparison of AUs and Pose between two datasets
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    bins = np.arange(0, 1.0, 0.05)

    # Overlaid histograms for AUs
    axes[0].set_xlim([min(au_intensity_std_means+compare_au_intensity_std_means)-0.4, max(au_intensity_std_means+compare_au_intensity_std_means)+0.4])
    axes[0].hist(au_intensity_std_means, bins=bins, alpha=0.5, label='UBFC-rPPG')
    axes[0].hist(compare_au_intensity_std_means, bins=bins, alpha=0.5, label='MAUBFC-rPPG')
    axes[0].set_title('Comparison of Mean Std. Dev. in AUs')
    axes[0].set_xlabel('Mean Std. Dev. AUs Intensity')
    axes[0].set_ylabel('# Videos')
    axes[0].legend(loc='upper right')

    # Overlaid histograms for Pose Rotations
    axes[1].set_xlim([min(pose_rotation_std_means+compare_pose_rotation_std_means)-0.2, max(pose_rotation_std_means+compare_pose_rotation_std_means)+0.2])
    axes[1].hist(pose_rotation_std_means, bins=bins, alpha=0.5, label='UBFC-rPPG')
    axes[1].hist(compare_pose_rotation_std_means, bins=bins, alpha=0.5, label='MAUBFC-rPPG')
    axes[1].set_title('Comparison of Mean Std. Dev. in Pose Rotations')
    axes[1].set_xlabel('Mean Std. Dev. Pose Rotations')
    axes[1].set_ylabel('# Videos')
    axes[1].legend(loc='upper right')

    # Adjust spacing between subplots
    fig.tight_layout()

    # Save the figure
    plt.savefig('ubfc_maubfc_compare_mean_std_AUs_and_Pose.png')
