#!/bin/bash
DIR=../../Data/videos/;
OUT=../../Results/SumMe/VSUMM/;
HOMEDIR=$PWD;
# choose pre-sampling rates and number of clusters for videos
# -1 for percent defaults to 1/100 of video length

# sampling rates for future use
# "1" "2" "5" "10" "25" "30"

# percent of the actual video
for method in "vsumm_gaussian"; do #"vsumm_kmeans" "vsumm_gaussian" "cnn_kmeans" "cnn_gaussian"; 
    for percent in "15"; do
    	for sampling_rate in "5"; do
    		for filename in $DIR"Base_jumping.mp4"; do
    			echo "parameters for vsumm feat"
    			echo $filename
    			echo $sampling_rate
    			cd $HOMEDIR
    			name=${filename##*/};
    			folder_name=${name%.mp4};
    			mkdir $OUT$folder_name;
    			# mkdir $OUT$folder_name"/keyframes";
    			echo $percent
    			echo $$OUT$folder_name"/"
    
    			echo "parameters for evaluation"
    			echo $filename
    			echo $sampling_rate
    			echo $percent
    			echo $OUT$folder_name"/" $OUT$folder_name"/Final_Results_cnn_reduced_"$percent".txt"
    			py allofshit.py $filename $sampling_rate $percent 0 0 1 $OUT$folder_name"/" $method;
                
    			cd ../Evaluation
    			py evaluate_vsumm.py $"$filename" $sampling_rate $percent "$OUT$folder_name/" "$OUT$folder_name/results_$method.txt" $method 1
    		done
    	done
    done
done