#!/bin/bash
DIR="../../Data/videos/"
OUT="../../Results/SumMe/SIFT/"
HOMEDIR="$PWD"

PYTHON2_PATH="C:/Python27/python"
SIFT_ANALYSIS="C:/Users/HP/py2env/vsumm/Scripts/SIFT/videoSumSIFT.py"
SIFT_EVALUATE="C:/Users/HP/py2env/vsumm/Scripts/SIFT/SIFTevaluate.py"

# Create output directory
mkdir -p "$OUT"

# Process all MP4 videos
for video in "$DIR"*.mp4; do
    name=$(basename "$video")
    folder_name="${name%.mp4}"
    
    echo "Processing $name..."
    
    # Create working directory
    work_dir="$OUT$folder_name"
    rm -rf "$work_dir"
    mkdir -p "$work_dir"
    
    # Extract frames
    mkdir -p "$work_dir/allFrames"
    ffmpeg -i "$video" "$work_dir/allFrames/image%05d.jpg" -hide_banner
    
    # Run SIFT analysis
    $PYTHON2_PATH $SIFT_ANALYSIS "$video" "$work_dir"
    
    # Evaluate results
    $PYTHON2_PATH $SIFT_EVALUATE "$video" 1 15 "$work_dir"
done

echo "Batch processing complete!"