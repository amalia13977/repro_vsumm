import numpy as np
import json
from tensorflow.keras.utils import get_file
import tensorflow.keras.backend as K

CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_data_format()

    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x


def decode_predictions(preds, top=5):
    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        fpath = get_file('imagenet_class_index.json',
                         CLASS_INDEX_PATH,
                         cache_subdir='models')
        CLASS_INDEX = json.load(open(fpath))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        results.append(result)
    return results




import numpy as np
import warnings

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import get_file
from keras import backend as K

TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5'
TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def VGG16(include_top=True, weights='imagenet',
          input_tensor=None):
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    if K.image_data_format() == 'th':
        if include_top:
            input_shape = (3, 224, 224)
        else:
            input_shape = (3, None, None)
    else:
        if include_top:
            input_shape = (224, 224, 3)
        else:
            input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)

    # Create model
    model = Model(img_input, x)

    # load weights
    if weights == 'imagenet':
        print('K.image_dim_ordering:', K.image_data_format())
        if K.image_data_format() == 'th':
            if include_top:
                weights_path = get_file('vgg16_weights_th_dim_ordering_th_kernels.h5',
                                        TH_WEIGHTS_PATH,
                                        cache_subdir='models')
            else:
                weights_path = get_file('vgg16_weights_th_dim_ordering_th_kernels_notop.h5',
                                        TH_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models')
            model.load_weights(weights_path)
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image dimension ordering convention '
                              '(`image_dim_ordering="th"`). '
                              'For best performance, set '
                              '`image_dim_ordering="tf"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
                convert_all_kernels_in_model(model)
        else:
            if include_top:
                weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                        TF_WEIGHTS_PATH,
                                        cache_subdir='models')
            else:
                weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        TF_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models')
            model.load_weights(weights_path)
            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)
    return model



HOMEVIDEOS='videos/'
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import ResNet50
from sklearn.decomposition import PCA
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

def get_vsumm_feat(frames_raw):
    frames = []
    pca = PCA(n_components=500)
    
    # Preprocess frames
    for im in frames_raw:
        im = cv2.resize(im, (224, 224)).astype(np.float32)
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        frames.append(np.expand_dims(im, axis=0))
    
    frames = np.array(frames)
    
    # Load VGG16 model
    base_model = VGG16(weights='imagenet', include_top=True)
    
    # Create feature extraction model
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
    
    # Extract features
    features = np.ndarray((frames.shape[0], 4096), dtype=np.float32)
    for i, x in enumerate(frames):
        features[i,:] = model.predict(x)
    
    return pca.fit_transform(features)

def get_color_hist(frames_raw, num_bins):
    print ("Generating linear Histrograms using OpenCV")
    channels=['b','g','r']
    
    hist=[]
    for frame in frames_raw:
        feature_value=[cv2.calcHist([frame],[i],None,[int(num_bins)],[0,256]) for i,col in enumerate(channels)]
        hist.append(np.asarray(feature_value).flatten())
    
    hist=np.asarray(hist)
    #print ("Done generating!")
    print ("Shape of histogram: " + str(hist.shape))
    
    return hist

def get_resnet_features(frames):
    # Load ResNet50 model (keep in memory for multiple calls)
    if not hasattr(get_resnet_features, 'model'):
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        get_resnet_features.model = Model(inputs=base_model.input, 
                                        outputs=base_model.output)
    
    features = []
    for frame in frames:
        # Convert BGR to RGB if needed (OpenCV uses BGR by default)
        if frame.shape[-1] == 3:  # BGR format
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:  # Handle grayscale or other formats
            img = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
        # Resize and preprocess
        img = cv2.resize(img, (224, 224))
        img = resnet_preprocess(img)  # Using imported function
        img = np.expand_dims(img, axis=0)
        
        # Extract features
        feat = get_resnet_features.model.predict(img, verbose=0)
        features.append(feat.flatten())
    
    return np.array(features)


import os
import sys
import numpy as np
import cv2
import scipy.io
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# frame chosen every k frames
sampling_rate=int(sys.argv[2])

# percent of video for summary
percent=int(sys.argv[3])

# globalizing
num_centroids=0

def save_keyframes(frame_indices, summary_frames):
    global sampling_rate, num_centroids
    if int(sys.argv[6])==1:
        print("Saving frame indices")
        method_name = sys.argv[8] if len(sys.argv) > 8 else "cnn_gaussian"
        out_file=open(f"{sys.argv[7]}frame_indices_{method_name}_{num_centroids}_{sampling_rate}.txt",'w')
        for idx in frame_indices:
            out_file.write(str(idx*sampling_rate)+'\n')
        print("Saved indices")

    if int(sys.argv[5])==1:
            print("Saving frames")
            for i,frame in enumerate(summary_frames):
                cv2.imwrite(str(sys.argv[7])+"keyframes/frame%d.jpg"%i, frame)
            print("Frames saved")

def main():
    global num_bins, sampling_rate, num_centroids, percent
    print("Opening video!")
    capture = cv2.VideoCapture(os.path.abspath(os.path.expanduser(sys.argv[1])))
    print("Video opened\nChoosing frames")
	
    #choosing the subset of frames from which video summary will be generateed
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
    frames = np.array(frames)#convert to (num_frames, width, height, depth)

    print("Frames chosen")
    print("Length of video %d" % frames.shape[0])
    
    # clustering: defaults to using the features
    print("Clustering")

    # Get method from command line
    method = sys.argv[8] if len(sys.argv) > 8 else "cnn_gaussian"

    # Feature extraction based on method
    if "cnn" in method:
        print ("CNN Check!")
        features = get_resnet_features(frames)
    else:  # vsumm methods
        print ("VSUMM Check!")
        features = get_vsumm_feat(frames)

    # converting percentage to actual number
    num_centroids=int(percent*frames.shape[0]*sampling_rate/100)   
    
	# choose number of centroids for clustering from user required frames (specified in GT folder for each video)
    if percent==-1:
        video_address=sys.argv[1].split('/')
        gt_file=video_address[len(video_address)-1].split('.')[0]+'.mat'
        video_address[len(video_address)-1]=gt_file
        video_address[len(video_address)-2]='GT'
        gt_file='/'.join(video_address)
        num_frames=int(scipy.io.loadmat(gt_file).get('user_score').shape[0])
    	# automatic summary sizing: summary assumed to be 1/10 of original video
        num_centroids=int(0.1*num_frames)

    if len(frames) < num_centroids:
        print("Samples too less to generate such a large summary")
        print("Changing to maximum possible centroids")
        num_centroids=frames.shape[0]

    # Clustering based on method
    if "kmeans" in method:
        print ("Kmeans Check!")
        cluster_model = KMeans(n_clusters=num_centroids).fit(features)
    else:
        print ("Gaussian Check!")
        cluster_model = GaussianMixture(n_components=num_centroids).fit(features)

    print("Done Clustering!")

    print("Generating summary frames")
    summary_frames = []

    if isinstance(cluster_model, GaussianMixture):
        # GaussianMixture version
        #probabilities = cluster_model.predict_proba(features)
        frame_indices = []
        for i in range(cluster_model.n_components):
            center = cluster_model.means_[i]
            dists = np.linalg.norm(features - center, axis=1)
            frame_indices.append(np.argmin(dists))

    else:
        # KMeans version
        distances = cluster_model.transform(features)
        frame_indices = [np.argmin(distances[:,cluster]) for cluster in range(cluster_model.n_clusters)]

    frame_indices = sorted(frame_indices)
    summary_frames = [frames[i] for i in frame_indices]
    print("Generated summary")
    
    if len(sys.argv)>5 and (int(sys.argv[5])==1 or int(sys.argv[6])==1):
        save_keyframes(frame_indices, summary_frames)
        print(f"Saved result for method: {method}")

if __name__ == '__main__':
    main()