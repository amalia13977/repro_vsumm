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
        gt_indicator = np.array([1 if x > 0 else 0 for x in user_score[:, userIdx]])
        user_intersection[userIdx] = np.sum(gt_indicator * summary_indicator)
        user_union[userIdx] = np.sum(np.array([1 if x > 0 else 0 for x in gt_indicator + summary_indicator]))
        user_length[userIdx] = np.sum(gt_indicator)
    
    recall=user_intersection/user_length;
    p=user_intersection/np.sum(summary_indicator);

    f_measure=[]
    for idx in range(0,len(p)):
        if p[idx]>0 or recall[idx]>0:
           f_measure.append(2*recall[idx]*p[idx]/(recall[idx]+p[idx]))
        else:
            f_measure.append(0)
    nn_f_meas=np.max(f_measure);
    f_measure=np.mean(f_measure);
    
    nnz_idx=np.nonzero(summary_selection)
    nbNNZ=len(nnz_idx[0])
         
    summary_length=float(nbNNZ)/float(len(summary_selection));
       
    recall=np.mean(recall);
    p=np.mean(p);
     
    return f_measure, summary_length


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



import sys
import os
import imageio
import cv2
# System Arguments
# Argument 1: Location of the video
# Argument 2: Sampling rate
# Argument 3: Percentage of video as summary
# Argument 4: Results folder


# Argument 5: File where the results will be written
# Argument 6: Name of the features used
# Argument 7: Skimming (Put 1)

def main():
    video = sys.argv[1]
    directory = sys.argv[4]
    sampling_rate = int(sys.argv[2])
    percent = int(sys.argv[3])
    method_name = sys.argv[6]  # Model name
    
    capture = cv2.VideoCapture(os.path.abspath(os.path.expanduser(sys.argv[1])))
    frames = []
    i=0
    while(capture.isOpened()):
        if i%sampling_rate==0:
            capture.set(1,i)
            # print i
            ret, frame = capture.read()
            if frame is None :
                break
            #im = np.expand_dims(im, axis=0) #convert to (1, width, height, depth)
            # print frame.shape
            frames.append(np.asarray(frame))
        i+=1
    frames = np.array(frames)
    num_centroids=int(percent*frames.shape[0]*sampling_rate/100)
    
    frame_file = os.path.join(directory, f"frame_indices_{method_name}_{num_centroids}_{sampling_rate}.txt")
    with open(frame_file, 'r') as f:
        frame_indices = [int(idx.strip()) for idx in f if idx.strip()]
    
    print("Got the frames'")
    
    video=video.split('/')
    videoName=video[len(video)-1].split('.')[0]
    print(videoName)
    
    video[len(video)-2]="GT"
    HOMEDATA='/'.join(video[0:len(video)-1])
    print(HOMEDATA)
    
    # OPTIONAL: Recreating summary
    # video=imageio.get_reader(sys.argv[1])
    # summary=np.array([video.get_data(idx) for idx in frame_indices])
    
    f_measure, summary_length=evaluateSummary(frame_indices,videoName,HOMEDATA)
    print("F-measure %.3f at length %.2f" %(f_measure, summary_length))
    if len(sys.argv)>5:
        if os.path.exists(sys.argv[5])==False:
            out_file=open(sys.argv[5],'a')
            out_file.write("Sampling rate, Number of Clusters, F-measure, Summary Length\n")
        else:
            out_file=open(sys.argv[5],'a')
        out_file.write("%d,%d,%f,%f\n"%(sampling_rate,num_centroids,f_measure,summary_length))
    
    # optional plotting of results
    #methodNames={'VSUMM using Color Histrograms'}
    #summaries={}
    #summaries[0]=frame_indices
    #plotAllResults(summaries,methodNames,videoName,HOMEDATA)

if __name__ == '__main__':
    main()