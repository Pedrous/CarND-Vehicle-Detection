[//]: # (Image References)
[image1]: ./output_images/output0.jpg
[image2]: ./output_images/output1.jpg
[image3]: ./output_images/output2.jpg
[image4]: ./output_images/output3.jpg
[image5]: ./output_images/output4.jpg
[image6]: ./output_images/output5.jpg
[image7]: ./examples/output_bboxes.png
[video1]: ./project_output.mp4

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 37 through 85 the python file `combined_classify.py`. I started by reading in all the `vehicle` and `non-vehicle` images.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I tried out various different parameters, different color spaces (RGB, HLS, HSV, LUV, YUV, YcrCb) and amount of orientations, pixels per cell and cells per block. Ultimately I chose following values, `orientation = 9`, `pix_per_cell = 8` and `cell_per_block = 2`.

#### 2. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using by using `sklearn.LinearSVC()` in lines 11 through 131 in file `combined_classify.py`, First I trained the classifier by using 80 % of the data to train and 20 % of the data to test the the classifier. The test accuracy was about 0.999 with randomly picked data and when I handpicked the data, the test accuracy was still more than 0.99. After this I also confirmed that there not too many false positives with this method in the test_images nor the video. Before this I normalized the data by using `sklearn.StandardScaler()`.

---
### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window technique is implemented in file `detectvideo.py` in lines 78 through 114. I decided to use for 2 cells for each step because it was giving me good enough results in comparison to 1 step for cell which was taking too much of time. The idea is that a preferred part of the image is selected, then it is scaled to preferred resolution and the HOG features are calculated for the whole image and after this eh sliding window technique is used to compare the features of the classifier to each 64 by 64 window in the image.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 4 scales (1, 1.25, 1.5 and 2) using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. I also used thresholding of the decision function to reduce the amount of false positives. Here are some example images (the problem is that I used the same pipeline for the test images than for the videos, but that obviously doesn't work because the are not consequent frames):

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

---
### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [video for my output](./project_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections over last 10 frames (This helped to reduce the noise and false postives as well) I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

---
### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are quite many parameters in the pipeline so it was really hard to find ones that work well and I was trying many different combinations before finding suitable ones for this exact problem. I guess that my pipeline could fail quite easily in other conditions because of too many false positives, the implementation was not tested for robustness. To improve the method, first of all the classifier should trained with much more data. Second of all, I would like to test the pipeline so that I would take the detections from various videos for each scale, let's say scales between 8 pixels and see how each scale can perform in each area. Then optimize the scales in the area to avoid unnecessary calculation and to optimize that the found boxes should be very close to the borders of the objects. After this it should be quite easy to filter the false positives by using the decision function and then the heatmap could be used to clarify the borders of the detected vehicles. Also the decision function values could be used to multiply the probabilities of the overlapping detections over few frames together to get an idea about if it is really a detection or not. When the measurements would be quite accurate, the estimate could be improved by using a kalman filter, although it might a bit complicated to define a model for the process. Also it would be nice to track the vehicle movement to know where to look for the vehicle in the next frame instead of searching the whole image in each frame. Then it would be enough to search for the borders and the vehicles that are already detected. 

