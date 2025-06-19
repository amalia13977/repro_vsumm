#!/bin/bash
DIR=../../Data/videos/;
OUT=../../Results/SumMe/Uniform_Sampling/;
HOMEDIR=$PWD;
sampling_rate="1";

# Update these paths to your Python 2.7 and scripts
PYTHON2_PATH="C:/Python27/python"  # Example: "C:/Python27/python"
UNIFORM_SCRIPT="$HOMEDIR/uniform.py"
EVALUATE_SCRIPT="C:/Users/HP/py2env/vsumm/Scripts/Evaluation/evaluate.py"

for percent in "15"; do
	echo $percent
	for filename in $DIR*".mp4";do
		echo $filename
		cd $HOMEDIR
		name=${filename##*/};
		folder_name=${name%.mp4};
		echo "printing previous shit"
		echo $folder_name
		mkdir $OUT$folder_name;
		echo $OUT$folder_name
		# Run uniform.py with full Python path
        $PYTHON2_PATH $UNIFORM_SCRIPT $filename $percent $OUT$folder_name"/"
		cd ../Evaluation
		pwd 
		echo "printing the stuff"
		echo $filename
		echo $sampling_rate
		echo $percent
		echo $OUT$folder_name"/" $OUT$folder_name"/final_results_uniform_"$percent".txt"
		$PYTHON2_PATH $EVALUATE_SCRIPT $filename $sampling_rate $percent $OUT$folder_name"/" $OUT$folder_name"/final_results_uniform_"$percent".txt" uniform;
	done
done