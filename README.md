# Human-Detection-HOG-LBP-Neural-Networks
Programs uses HOG (Histograms of Oriented Gradients) and LBP (Local Binary Pattern) features to detect human in images. 
1. First, use the HOG feature only to detect humans.
2. Next, combine the HOG feature with the LBP feature to form an augmented feature (HOG-LBP) to detect human. 
3. A Two-Layer Perceptron (feedforward neural network) will be used to classify the input feature vector into human or no-human.  

## Steps

### Conversion to grayscale
The inputs to your program are color sub-images cut out from larger images. First, convert the color images into grayscale using the formula I=Round(0.299R+0.587G+0.114B) where R, G and B are the pixel values from the red, green and blue channels of the color image, respectively, and Round is the round off operator. 

### Gradient operator
Use the Sobel’s operator for the computation of horizontal and vertical gradients. Use formula M(i,j)=√(G_x^2+G_y^2 ) to compute gradient magnitude, where G_x  and G_y are the horizontal and vertical gradients. Normalize and round off the results to integers within the range [0, 255]. Next, compute the gradient angle (with respect to the positive x axis that points to the right.) For image locations where the templates go outside of the borders of the image, assign a value of 0 to both gradient magnitude and gradient angle. Also, if both G_x  and G_y are 0, assign a value of 0 to both gradient magnitude and gradient angle. 

### HOG feature
Refer to the lecture slides for the computation of the HOG feature. Use the unsigned representation and quantize the gradient angle into one of the 9 bins as shown in the table below. If the gradient angle is within the range [170, 350), simply subtract by 180 first. Use the following parameter values in your implementation: cell size = 8 x 8 pixels, block size = 16 x 16 pixels (or 2 x 2 cells), block overlap or step size = 8 pixels (or 1 cell.)  Use L2 norm for block normalization. Leave the histogram and final feature values as floating point numbers. Do not round off to integers. 
		
### Histogram Bins
Bin #	Angle in degrees	Bin center
1	     [-10,10)	            0
2	     [10,30)	            20
3	     [30,50)	            40
4	     [50,70)	            60
5	     [70,90)	            80
6	     [90,110)	            100
7	     [110,130)	          120
8	     [130,150)	          140
9	     [150,170)	          160

### LBP feature
For the computation of the LBP feature, first divide the input image into non-overlapping blocks of size 16×16.  Next, compute LBP patterns (refer to lecture slides) at each pixel location inside the blocks and convert the 8-bit patterns into decimals within the range [0, 255]. Then, form a histogram of the LBP patterns for each block. To reduce the dimension of the histogram, we create separate bins for uniform patterns and a single bin for all non-uniform patterns.   An 8-bit LBP pattern is called uniform if the binary pattern contains at most two 0-1 or 1-0 transitions if we go around the pattern in circle. For example, 00010000 (2 transitions) is a uniform pattern, but 01010100 (6 transitions) is not. By putting all non-uniform patterns into a single bin, the dimension of the histogram is reduced from 256 to 59. The 58 uniform binary patterns correspond to the integers 0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254 and 255, and all other integers belong to non-uniform patterns. Let the 1st to 58th bins of your histogram be assigned to the uniform patterns according to the order above, and the 59th bin be assigned to non-uniform patterns. For pixels in the first and last rows, and first and last columns of the image, we cannot compute their LBP patterns since some of their 8-neigbors are outside of the borders of the image. Simply assign a LBP value of 5 at these pixel locations and they will be assigned to the 59th bin of the histogram for non-uniform patterns. Finally, concatenate the histograms from all blocks (in left to right, then top to bottom order) to form a single feature vector. 

### HOG-LBP feature
To form the combined HOG-LBP feature, simply concatenate the HOG and LBP feature vectors together to form a long vector.

### Two-layer perceptron
Implement a fully-connected two-layer perceptron with an input layer of size N, with N being the size of the input feature vector, a hidden layer of size H and an output layer of size 1. Let H=200 and 400 and report the training and classification results for each. (Optional: you can try other hidden layer sizes and report the results if you get better results than the two above.) Use the ReLU activation function for neurons in the hidden layer and the Sigmoid function for the output neuron. The Sigmoid function will ensure that the output is within the range [0,1], which can be interpreted as the probability of having detected human in the image. Use the weight updating rules we covered in lecture for the training of the two-layer perceptron. Use random initialization to initialize the weights of the perceptron. Assign an output label of 1.0 for training images containing human and 0.0 for training images with no human. You can experiment with and decide on the learning rate to use (can try 0.1 first.) After each epoch of training, compute the average error from the errors of individual training samples. The error for an individual training sample =|correct output-network output|, with the correct output equals 1.0 for positive samples and 0.0 for negative samples. You can stop training when the change in average error between consecutive epochs is less than some threshold (e.g., 0.1) or when the number of epochs is more than some maximum (e.g., 1000.) After training, you can use the perceptron to classify the test images. Use the following rules for classification:

### Perceptron Output Classification
≥0.6 - human

>0.4 and <0.6 - borderline

≤0.4 - no-human

### Training and test images
A set of 20 training images and a set of 10 test images in .bmp format will be provided. The training set contains 10 positive (human) and 10 negative (no human) samples and the test set contains 5 positive and 5 negative samples. All images are of size 160 (height) X 96 (width).

### Experiments
Perform experiments with hidden layer sizes of 200 and 400 in the perceptron, and for each hidden layer size, use the HOG only feature and then the combined HOG-LBP feature (a total of four experiments.) 

(a) HOG only feature. Given the image size of 160×96 and the parameters given above for HOG computation, you should have 20 X 12 cells and 19 X 11 blocks. The size of your feature vector (and the size of the input layer of your perceptron) is therefore 7,524. 

(b) Combined HOG-LBP feature. With the parameters given above for the LBP feature, there are 10×6 blocks in the input image and the size of the LBP feature is 10×6×59=3,540. The combined HOG-LBP feature therefore has size 7524+3540=11,064.
