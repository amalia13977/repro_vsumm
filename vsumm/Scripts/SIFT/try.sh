#!/bin/bash
PYTHON2_PATH="C:/Python27/python"
pwd=dir/
name=$1;
folder_name=${name%.mp4};
allFrames=allFrames;
keyFrames=keyFrames;
rm -r $folder_name"/"
mkdir $folder_name
cd $folder_name		#pwd=dir/$folder_name
mkdir $allFrames
cd $allFrames		#pwd=dir/$folder_name/$allFrames
ffmpeg -i ../../../../Data/videos/$name image%d.jpg
mkdir ../$keyFrames
cd ../../			#pwd=dir/
# Run SIFT analysis
$PYTHON2_PATH $SIFT_ANALYSIS "$video" "$work_dir"
    
# Evaluate results
$PYTHON2_PATH $SIFT_EVALUATE "$video" 1 15 "$work_dir"