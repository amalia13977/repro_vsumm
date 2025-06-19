#!/bin/bash
#unzip SumMe.zip
#rm -rf SumMe.zip

# converts space separated file names to underscore separated names
cd C:/Users/HP/Documents/BISMILLAH-REPRODUCE/Mahmood-Jasim/Video-Summarization-using-Keyframe-Extraction-and-Video-Skimming-master/Data/videos
for file in *; do mv "$file" `echo $file | tr ' ' '_'` ; done

cd C:/Users/HP/Documents/BISMILLAH-REPRODUCE/Mahmood-Jasim/Video-Summarization-using-Keyframe-Extraction-and-Video-Skimming-master/Data/GT
for file in *; do mv "$file" `echo $file | tr ' ' '_'` ; done