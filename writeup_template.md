## Writeup Template
---

** Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/vehicle_and_non_vehicle.png
[image2]: ./examples/hog_example1.png
[image3]: ./examples/sliding_window_entire_image.png
[image4]: ./examples/sliding_window_ycropped_image.png
[image5]: ./examples/heatmap1.png
[image6]: ./examples/label_on_heatmap1.png
[image7]: ./examples/output_bboxes.png
[image8]: ./examples/hog_example2.png
[image9]: ./examples/heatmap2.png
[image10]: ./examples/heatmap3.png
[image11]: ./examples/label_on_heatmap2.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 1st code cell of the IPython notebook `CarND-Vehicle-Detection.ipynb` from lines 13 through 27 in a method called `get_hog_features`. A method named `extract_features` in line 54 of cell 1 takes the image, hog parameters, spatial and histogram feature flags and color space as parameters. It eventually concatenates and return an array of hog features for each image by calling the `get_hog_features` method and the other methods based on the paramters passed to the  `extract_features` method.    

After defining the helper functions I started by reading all the `vehicle` and `non-vehicle` images. This is done in the 2nd code cell from lines 9 through 41.  I have taken a screenshot of the output from the Jupyter notebook of 5 random car and non-car images. Here is an example of `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different HOG parameters such as `color_space`,       `orient` and `pixels_per_cell`.  I grabbed random images from each of the two classes and displayed them to get a feel of what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orient=6`, `pixels_per_cell=8` and `cells_per_block=2`:


![alt text][image2]

Here is another example using the `YCrCb` color space and HOG parameters of `orient=9`, `pixels_per_cell=8` and `cells_per_block=2`:

![alt text][image8]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. Below are few examples:

#### Combination 1      
##### Parameters
color_space = 'RGB'   
orient = 6     
pix_per_cell = 8   
cell_per_block = 2       
hog_channel = 0         

##### Results
53.290613889694214 Seconds to complete features...     
Using: 6 orientations 8 pixels per cell 2 cell per block 32 histogram bins, and (32, 32) spatial sampling       
Feature vector length: 4344        
15.01 Seconds to train SVC...       
Test Accuracy of SVC = 0.9702

#### Combination 2      
##### Parameters   
color_space = 'YCrCb'       
orient = 9        
pix_per_cell = 8      
cell_per_block = 2      
hog_channel = 0        

##### Results
43.361576080322266 Seconds to complete features...        
Using: 9 orientations 8 pixels per cell 2 cell per block 32 histogram bins, and (32, 32) spatial sampling         
Feature vector length: 4932         
13.44 Seconds to train SVC...       
Test Accuracy of SVC = 0.9865

#### Combination 3         
##### Parameters 
color_space = 'YCrCb'        
orient = 9        
pix_per_cell = 8       
cell_per_block = 2       
hog_channel = 'ALL'         

##### Results        
84.96577596664429 Seconds to complete features...         
Using: 9 orientations 8 pixels per cell 2 cell per block 32 histogram bins, and (32, 32) spatial sampling        
Feature vector length: 8460        
23.96 Seconds to train SVC...         
Test Accuracy of SVC = 0.9932

I was monitoring the feature extraction time, train time and test accuracy in each combination. For this project i decided to go with the model which has the highest accuracy. In a production grade project i would build a matrix of permuations and combinations of all the parameters, feature extraction time and result accuracy and depending on the project needs i would pick up the desired combination. 

I chose the 3rd combincation from the above examples as the time taken to extract the features is reasonable.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In cell 4 of the IPython notebook `CarND-Vehicle-Detection.ipynb` in line 60 i use SVC classifier for training the model. I passed the followig parameters to the `extract_features` method :

color_space = 'YCrCb'     
orient = 9      
pix_per_cell = 8      
cell_per_block = 2      
hog_channel = 'ALL'      
spatial_size = (32, 32)             
hist_bins = 32        
spatial_feat = True          
hist_feat = True             
hog_feat = True


The `extract_features` method returns the feature vectors of the cars and non-cars which i then stack up using the numpy's vstack method and then covert it to float. I then label the 1's for the cars and 0's for the non-cars. Then i split the data to training and test set and train the model.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I started experimenting with the sliding window in cell 5 of the IPython notebook `CarND-Vehicle-Detection.ipynb`. I first applied the sliding window without specifying the Y coordinates, window overlap of 50% and chose 128X128 window size. Below is the output with the specified parameters:

![alt text][image3]

As it is evident from the image i found quite many false positives especially in areas of the image where we typically dont find cars. I further continued experimenting with various window sizes such as 64X64 an 96X96 and window overlaps ratios and finally settled with a window overlap ratio of 50%, Y dimension of 400,656  and a window size of 96X96. Below is the output with these parameters:

![alt text][image4]

However, the above approach slows down the pipeline as it takes roughly half a second to process one image with my previous setting. So instead of extracting features from each window individually i took the HOG feature of the entire image once and then sub-sampled that array to extract the features for each window.




#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I initially experimented by extracting only the Y channel or YCrCb color space, orientation of 6 and spatial size of (16,16). At the end I searched on All the 3-channels of YCrCb, HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image6]
![alt text][image11]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_result.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in the example images in the test_images folder.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of example images and the result of `scipy.ndimage.measurements.label()`:

### Here are eight frames and their corresponding heatmaps:

![alt text][image5]
![alt text][image9]
![alt text][image10]


### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all 8 frames:
![alt text][image6]
![alt text][image11]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The first problem i faced was an issue with the vector length when i was testing the example images, i received the following error: ValueError: operands could not be broadcast together with shapes (1,8460) (3168,) (1,8460). After tracing back each function call i was able to fix the bug which turned out to be a silly indentation issue due to which the array was not appending the data points as desired.The next tricky part was experimenting various HOG parameters and color spaces and choosing the ideal combination. 

The pipeline might likely fail if there there are several parallel cars with the same color next to each other or if the car has a similar color as the background. In addition if there are cars which are unique in their shape and size and if such an example is not available in the training set, the pipeline could still fail.

I would may be augument my dataset with more car and non-car examples and try out a convolution neural network to train my model to overcome the above problems.

##### Referecenes:

I used the Udacity classroom material and [Udacity Q&A](https://www.youtube.com/watch?v=P2zwrTM8ueA&index=5&list=PLAwxTw4SYaPkz3HerxrHlu1Seq8ZA7-5P) as my references.





