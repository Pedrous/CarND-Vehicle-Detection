import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pdb
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from functions import *


# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features
    

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features
    

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features
        

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, orient=9, pix_per_cell=8, cell_per_block=2, spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256)):
	# Create a list to append feature vectors to
	features = []
	# Iterate through the list of images
	for file in imgs:
		# Read in each one by one
		img = cv2.imread(file)
	
		ctrans_tosearch = convert_color(img, conv='BGR2YCrCb')
		#images = [ctrans_tosearch, cv2.flip(ctrans_tosearch, 1)]
			
		#for image in images:
		ch1 = ctrans_tosearch[:,:,0]
		ch2 = ctrans_tosearch[:,:,1]
		ch3 = ctrans_tosearch[:,:,2]
	
		# Compute individual channel HOG features for the entire image
		hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=True)
		hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=True)
		hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=True)
		hog_features = np.hstack((hog1, hog2, hog3))
	
		# Get color features
		spatial_features = bin_spatial(ctrans_tosearch, size=spatial_size)
		hist_features = color_hist(ctrans_tosearch, nbins=hist_bins)
	
		# Scale features and make a prediction
		features.append(np.concatenate((spatial_features, hist_features, hog_features)))#.reshape(1, -1))
		# features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
		# test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
				
	return features


# Divide up into cars and notcars
images = glob.glob('*vehicles/**/*.png')
cars = []
notcars = []
for image in images:
    if 'non-vehicles' in image:
        notcars.append(image)
    else:
        cars.append(image)

### TODO: Tweak these parameters and see how the results change.
orient = 9 #16
pix_per_cell = 8 #16
cell_per_block = 2
spatial = 32 #12
histbin = 32 #4

t=time.time()
car_features = extract_features(cars, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=(spatial, spatial), hist_bins=histbin, hist_range=(0, 256))
notcar_features = extract_features(notcars, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=(spatial, spatial), hist_bins=histbin, hist_range=(0, 256))
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler

X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
#rand_state = np.random.randint(0, 100)
#X_train, X_test, y_train, y_test = train_test_split(
#    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(scaled_X, y)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
#print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
#t=time.time()
#n_predict = 10
#print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
#print('For these',n_predict, 'labels: ', y_test[0:n_predict])
#t2 = time.time()
#print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

#Save the model and parameters
obj = dict(((k, eval(k)) for k in ('orient', 'pix_per_cell', 'cell_per_block', 'spatial', 'histbin', 'svc', 'X_scaler')))
with open('classifier.p', 'wb') as handle:
    pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
