#!/bin/bash
export OMP_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4

# Update this path to point to a dataset folder of MP4 videos you'd like to analyze
input_dir="/path/to/converted_mp4_videos"
# Update this path to point to a folder to hold OpenFace outputs
output_dir="/path/to/OpenFace/analysis/output"
# Update this path to point to the FeatureExtraction executable after installing OpenFace
openface_dir="/path/to/OpenFace/installation/OpenFace/build/bin/FeatureExtraction"

# Iterate over files in the input directory
for i in "$input_dir"/*; do
    # Extract the base filename
    filename=$(basename "$i")
    echo "$filename"

    # Run the OpenFace command
    "$openface_dir" -f "$i" -pose -aus -2Dfp -3Dfp -pdmparams -out_dir "$output_dir" -of "$filename"
done
