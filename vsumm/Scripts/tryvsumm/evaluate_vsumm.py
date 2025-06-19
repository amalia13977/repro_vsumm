import scipy.io
import warnings
import numpy as np
import matplotlib.pyplot as plt

def evaluateSummary(summary_selection, videoName, HOMEDATA):
    '''Evaluates a summary for video videoName'''
    # Convert input to list if it's a map object
    if isinstance(summary_selection, map):
        summary_selection = list(summary_selection)

    # Load GT file
    gt_file = os.path.join(HOMEDATA, f'{videoName}.mat')
    gt_data = scipy.io.loadmat(gt_file)

    user_score = gt_data.get('user_score')
    nFrames = user_score.shape[0]
    nbOfUsers = user_score.shape[1]

    # Convert and pad summary selection
    if len(summary_selection) < nFrames:
        summary_selection = np.pad(summary_selection, (0, nFrames-len(summary_selection)), 'constant')
    elif len(summary_selection) > nFrames:
        summary_selection = summary_selection[:nFrames]

    summary_indicator = np.array([1 if x > 0 else 0 for x in summary_selection])
    # Initialize arrays
    user_intersection = np.zeros(nbOfUsers, dtype=np.float32)
    user_union = np.zeros(nbOfUsers, dtype=np.float32)
    user_length = np.zeros(nbOfUsers, dtype=np.float32)

    # Calculate metrics per user
    for userIdx in range(nbOfUsers):
        gt_indicator = np.array([1 if x > 0 else 0 for x in gt_data['user_score'][:, userIdx]]) # Corrected: access user_score for current user
        user_intersection[userIdx] = np.sum(np.logical_and(summary_indicator, gt_indicator))
        user_union[userIdx] = np.sum(np.logical_or(summary_indicator, gt_indicator))
        user_length[userIdx] = np.sum(gt_indicator)

    # Calculate precision, recall, and F-measure
    precision = user_intersection / np.sum(summary_indicator) if np.sum(summary_indicator) > 0 else 0
    recall = user_intersection / user_length if np.sum(user_length) > 0 else 0

    # Avoid division by zero for F-measure
    f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Average over users
    avg_f_measure = np.mean(f_measure)
    avg_summary_length = np.mean(np.sum(summary_indicator)) # This seems to be the total length, not percentage

    # The paper's F-measure calculation might be different, double-check it.
    # This implementation calculates F-measure for each user and then averages.
    # Often, precision and recall are calculated globally across all ground truths.

    return avg_f_measure, avg_summary_length

def plotAllResults(summary_selections,methods,videoName,HOMEDATA):
    '''Evaluates a summary for video videoName and plots the results
      (where HOMEDATA points to the ground truth file) 
      NOTE: This is only a minimal version of the matlab script'''
    
    # Get GT data
    gt_file=HOMEDATA+'/'+videoName+'.mat'
    gt_data = scipy.io.loadmat(gt_file)
    user_score=gt_data.get('user_score')
    nFrames=user_score.shape[0];
    nbOfUsers=user_score.shape[1];    

    ''' Get automated summary score for all methods '''
    automated_fmeasure={};
    automated_length={};
    for methodIdx in range(0,len(methods)):
        summaryIndices=np.sort(np.unique(summary_selections[methodIdx]))
        automated_fmeasure[methodIdx]=np.zeros(len(summaryIndices));
        automated_length[methodIdx]=np.zeros(len(summaryIndices));
        idx=0
        for selIdx in summaryIndices:
            if selIdx>0:
                curSummary=np.array(map(lambda x: (1 if x>=selIdx else 0),summary_selections[methodIdx]))    
                f_m, s_l = evaluateSummary(curSummary,videoName,HOMEDATA)
                automated_fmeasure[methodIdx][idx]=f_m
                automated_length[methodIdx][idx]=s_l
                idx=idx+1

    
    ''' Compute human score '''
    human_f_measures=np.zeros(nbOfUsers)
    human_summary_length=np.zeros(nbOfUsers)
    for userIdx in range(0, nbOfUsers):
        human_f_measures[userIdx], human_summary_length[userIdx] = evaluateSummary(user_score[:,userIdx],videoName,HOMEDATA);

    avg_human_f=np.mean(human_f_measures)
    avg_human_len=np.mean(human_summary_length)
    

    ''' Plot results'''
    fig = plt.figure()
    p1=plt.scatter(100*human_summary_length,human_f_measures)
    colors=['r','g','m','c','y']
    for methodIdx in range(0,len(methods)):
        p2=plt.plot(100*automated_length[methodIdx],automated_fmeasure[methodIdx],'-'+colors[methodIdx])
        
    plt.xlabel('summary length[%]')
    plt.ylabel('f-measure')
    plt.title('f-measure for video '+videoName)
    legend=list(methods)    
    legend.extend(['individual humans'])
    plt.legend(legend)
    plt.ylim([0,0.85])
    plt.xlim([0,20])
    plt.plot([5, 5],[0, 1],'--k')
    plt.plot([15.1, 15.1],[ 0, 1],'--k')
    plt.show()



import scipy.io
import warnings
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import imageio # Make sure imageio is imported

# ... (rest of evaluateSummary function) ...

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    if len(sys.argv) < 6:
        print("Usage: python evaluate_vsumm.py <video_path> <sampling_rate> <percent> <output_directory> <method_name> [output_log_file]")
        sys.exit(1)

    video_path = sys.argv[1]
    sampling_rate = int(sys.argv[2])
    percent = int(sys.argv[3])
    directory = sys.argv[4]
    method_name = sys.argv[5]

    # --- MODIFICATION START ---
    # Instead of loading all frames, just get the number of frames (nFrames)
    # You can often get this from video metadata without loading all data.
    try:
        video_reader = imageio.get_reader(video_path)
        metadata = video_reader.get_meta_data()
        nFrames = metadata.get('nframes')
        if nFrames is None: # Fallback if 'nframes' isn't directly available in metadata
            print("Warning: 'nframes' not directly in metadata. Iterating to count frames (may be slow for large videos).")
            nFrames = 0
            for _ in video_reader:
                nFrames += 1
            # Re-open reader as it might be exhausted
            video_reader = imageio.get_reader(video_path)
            
    except Exception as e:
        print(f"Error getting video metadata or counting frames: {e}")
        print("Attempting to get frame count by iterating (might still be memory-intensive if frame objects are large even without numpy.array conversion).")
        nFrames = 0
        try:
            # Re-open the reader if it was partially read
            video_reader = imageio.get_reader(video_path)
            for _ in video_reader:
                nFrames += 1
            video_reader.close() # Close the reader after counting
        except Exception as e:
            print(f"Could not get frame count: {e}")
            sys.exit(1)

    if nFrames is None or nFrames == 0:
        print(f"Could not determine the number of frames for {video_path}. Exiting.")
        sys.exit(1)
    
    print(f"Total number of frames (nFrames): {nFrames}")

    # Now use nFrames instead of frames.shape[0]
    num_centroids = int(percent * nFrames * sampling_rate / 100)
    # --- MODIFICATION END ---

    frame_file = os.path.join(directory, f"frame_indices_{method_name}_{num_centroids}_{sampling_rate}.txt")
    print(f"Attempting to open frame_file: {frame_file}")

    if not os.path.exists(frame_file):
        print(f"Error: Frame index file not found at {frame_file}. Please check allofshit.py output and paths.")
        sys.exit(1)

    with open(frame_file, 'r') as f:
        frame_indices = [int(idx.strip()) for idx in f if idx.strip()]

    print("Got the frames' indices from file.") # Changed print message

    video_parts = video_path.split('/') # Use video_parts to avoid conflict with `video_reader` object
    videoName = video_parts[len(video_parts)-1].split('.')[0]
    print(f"Video Name: {videoName}")

    # Adjust HOMEDATA path as discussed in previous response
    # This assumes 'videos' and 'GT' are siblings within 'Data' folder structure
    # e.g., ../../Data/videos/video.mp4 -> ../../Data/GT/video.mat
    # It might need further adjustment based on your exact file structure
    HOMEDATA_parts = video_parts[0:len(video_parts)-2] # Get path up to 'Data'
    HOMEDATA = os.path.join('/'.join(HOMEDATA_parts), "GT")
    print(f"HOMEDATA (Ground Truth Directory): {HOMEDATA}")

    f_measure, summary_length = evaluateSummary(frame_indices, videoName, HOMEDATA)
    print("F-measure %.3f at length %.2f" % (f_measure, summary_length))

    if len(sys.argv) > 6:
        output_log_file = sys.argv[6]
        # Ensure directory exists for the log file
        os.makedirs(os.path.dirname(output_log_file), exist_ok=True)
        
        if not os.path.exists(output_log_file):
            out_file = open(output_log_file, 'a')
            out_file.write("Sampling rate, Number of Clusters, F-measure, Summary Length, Method Name\n")
        else:
            out_file = open(output_log_file, 'a')
        
        out_file.write(f"{sampling_rate},{num_centroids},{f_measure:.3f},{summary_length:.2f},{method_name}\n")
        out_file.close()