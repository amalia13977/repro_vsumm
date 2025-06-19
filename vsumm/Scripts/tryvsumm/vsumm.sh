#!/bin/bash
DIR=../../Data/videos/;
OUT=../../Results/SumMe/VSUMM/;
HOMEDIR=$PWD;

for method in "cnn_gaussian"; do # Add other methods here, e.g., "vsumm_kmeans" "vsumm_gaussian" "cnn_kmeans"
    for percent in "15"; do
        for sampling_rate in "5"; do
            for filename in $DIR"Air_Force_One.mp4"; do
                echo "parameters for vsumm feat"
                echo $filename
                echo $sampling_rate
                cd $HOMEDIR
                name=${filename##*/};
                folder_name=${name%.mp4};
                mkdir -p $OUT$folder_name; # Use -p to create parent directories if they don't exist
                echo $percent
                echo "$OUT$folder_name/"

                echo "parameters for evaluation"
                echo $filename
                echo $sampling_rate
                echo $percent
                echo "$OUT$folder_name/" "$OUT$folder_name/Final_Results_cnn_reduced_"$percent".txt"
                # Call allofshit.py
                py allofshit.py "$filename" "$sampling_rate" "$percent" 0 0 1 "$OUT$folder_name/" "$method";

                # Pass the 'method' as an argument to evaluate_vsumm.py
                py evaluate_vsumm.py "$filename" "$sampling_rate" "$percent" "$OUT$folder_name/" "$method" "$OUT$folder_name/Final_Results_${method}_reduced_"$percent".txt";
                cd $HOMEDIR # Go back to HOMEDIR after evaluation
            done
        done
    done
done