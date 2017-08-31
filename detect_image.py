import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
import pdb
from functions import *
from scipy.ndimage.measurements import label
from collections import deque

with open('classifier.p', 'rb') as handle:
    dist_pickle = pickle.load(handle)
 
svc = dist_pickle["svc"]
X_scaler = dist_pickle["X_scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = (dist_pickle["spatial"], dist_pickle["spatial"])
hist_bins = dist_pickle["histbin"]

images = glob.glob('test_images/*.jpg')
imgs = []
for image in images:
	imgs.append(mpimg.imread(image))
	

# Define a class to receive the characteristics of each line detection
class Detection():
	def __init__(self):
		self.last10det = deque(maxlen=10)
		self.continuity = deque(maxlen=2)
		self.boxes = deque(maxlen=15)
		
	def addcontframes(self, box):
		#pdb.set_trace()
		if ((len(box) == 0) & (len(self.continuity) > 0)):
			self.continuity.popleft()
		else:
			self.continuity.extend(box)


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, xstart, xstop, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
	draw_img = np.copy(img)
	#img = img.astype(np.float32)/255

	img_tosearch = img[ystart:ystop,xstart:xstop,:]
	ctrans_tosearch = convert_color(img_tosearch, conv='BGR2YCrCb')
	if scale != 1:
		imshape = ctrans_tosearch.shape
		ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
		
	ch1 = ctrans_tosearch[:,:,0]
	ch2 = ctrans_tosearch[:,:,1]
	ch3 = ctrans_tosearch[:,:,2]

	# Define blocks and steps as above
	nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
	nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
	nfeat_per_block = orient*cell_per_block**2

	# 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
	window = 64
	nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
	cells_per_step = 2  # Instead of overlap, define how many cells to step
	nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
	nysteps = (nyblocks - nblocks_per_window) // cells_per_step

	# Compute individual channel HOG features for the entire image
	hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
	hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
	hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
	
	bboxes = []

	for xb in range(nxsteps):
		for yb in range(nysteps):
			ypos = yb*cells_per_step
			xpos = xb*cells_per_step
			# Extract HOG for this patch
			hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

			xleft = xpos*pix_per_cell
			ytop = ypos*pix_per_cell

			# Extract the image patch
			subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
		  
			# Get color features
			spatial_features = bin_spatial(subimg, size=spatial_size)
			hist_features = color_hist(subimg, nbins=hist_bins)

			# Scale features and make a prediction
			test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
			#test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
			test_prediction = svc.predict(test_features)
			#pdb.set_trace()
			test_confidence = svc.decision_function(test_features)
			
			if ((test_prediction == 1) & (test_confidence[0] >= 0.4)):
				xbox_left = np.int(xleft*scale)
				ytop_draw = np.int(ytop*scale)
				win_draw = np.int(window*scale)
				bboxes.append(((xbox_left+xstart,ytop_draw+ystart),(xbox_left+win_draw+xstart, ytop_draw+win_draw+ystart)))
				cv2.rectangle(draw_img,(xbox_left+xstart, ytop_draw+ystart),(xbox_left+win_draw+xstart,ytop_draw+win_draw+ystart),(0,0,255),3)
				string = '{}'.format(round(test_confidence[0],2))
				cv2.putText(draw_img, string, (xbox_left+xstart+10,ytop_draw+ystart+60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)

	return bboxes, draw_img


def add_heat(heatmap, bbox_list):
  	# Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    bboxes = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 3)
    # Return the image
    return img, bboxes


def ScaleDetections(img):
	scales = [1, 1.25, 2, 1.5, 1.5]
	ystart = [400, 400, 464]
	ystop = [480, 528, 656]
	xcen = int(1280/2)
	widths = [704, 1280, 1280]
	corners = []
	for ytop,ybot,w in zip(ystart, ystop, widths):
		corners.append(((xcen-int(w/2), ytop), (xcen+int(w/2), ybot)))
		
	corners.append(((1054, 400),(1280, 592)))
	corners.append(((0, 400),(226, 592)))

	box_list = []
	frame_detections = np.copy(img)
	for corner, scale in zip(corners, scales):
		xstart = corner[0][0]
		xstop = corner[1][0]
		ystart = corner[0][1]
		ystop = corner[1][1]
		boxes, frame_detections = find_cars(frame_detections, xstart, xstop, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
		
		box_list.extend(boxes)
	
	edge_img = draw_boxes(img, corners)
		
	return edge_img, frame_detections, box_list
	

def HeatMap(img, box_list, thresh=2):
	heat = np.zeros_like(img[:,:,0]).astype(np.float)
	# Add heat to each box in box list
	heat = add_heat(heat,box_list)
	# Apply threshold to help remove false positives
	heat = apply_threshold(heat, min(thresh, max([4, 1/2*np.max(heat)])))
	# Visualize the heatmap when displaying    
	heat = np.clip(heat, 0, 255)
	normed_heat = (heat - np.min(heat)) / (np.max(heat) - np.min(heat))
	cmap = plt.get_cmap('hot')
	heatmap = cmap(normed_heat)
	heatmap = np.delete(heatmap, 3, 2)
	heatmap = np.uint8(heatmap*255)
	# Find final boxes from heatmap using label function
	labels = label(heat)
	
	return heatmap, labels


def Pipeline(img):
	# Draw the dimensions for the search areas and get the detections for the frame
	edge_img, frame_detections, box_list = ScaleDetections(img)
	#Save boxes for the last 15 detections
	detect.last10det.append(box_list)
	img_last10 = np.copy(img)
	#Draw the boxes for the last 15 detections
	boxes_list = []
	for bboxes in detect.last10det:
		img_last10 = draw_boxes(img_last10, bboxes)
		boxes_list.extend(bboxes)
	# Add heat and get labels for the last 15 detections
	last10_heatmap, labels = HeatMap(img, boxes_list, thresh = 5)
	# Get the detected vehicles after adding heat
	last10_detection, bboxes = draw_labeled_bboxes(np.copy(img), labels)

	string1 = '1. Areas for different scales'
	string2 = '2. Detections for the current frame'
	string3 = '3. Detections in the last 10 frames, amount of boxes: {}'.format(len(boxes_list))
	string4 = '4. Added heat for the detections in the last 10 frames'
	string5 = '5. Overall estimate of the vehicles'
	
	out1 = cv2.putText(edge_img, string1, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3, cv2.LINE_AA)
	out2 = cv2.putText(frame_detections, string2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3, cv2.LINE_AA)
	out3 = cv2.putText(img_last10, string3, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3, cv2.LINE_AA)
	out4 = cv2.putText(last10_heatmap, string4, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3, cv2.LINE_AA)
	out5 = cv2.putText(last10_detection, string5, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3, cv2.LINE_AA)
	
	comb_out1 = np.hstack((out1, out2, out3))
	comb_out2 = np.hstack((out4, out5, np.zeros_like(out5)))
	comb_out = np.vstack((comb_out1, comb_out2))
	out = cv2.resize(comb_out, (1920, 720))
	return out


### The pipeline for handling the video ###
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

detect = Detection()

images = glob.glob('test_images/*.jpg')
imgs = []
i = 0
for image in images:
	file = 'output_images/output{}.jpg'.format(i)
	cv2.imwrite(file, Pipeline(cv2.imread(image)))
	i += 1