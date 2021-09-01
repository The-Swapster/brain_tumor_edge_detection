# brain_tumor_edge_detection

## Dataset
The dataset used was from kaggle 
https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection

## Preprocessing
* Creating a function for augmenting the dataset
  1. Rotating range 10
  2. Width shift range of 0.1
  3. Height shift range of 0.1
  4. Shear range 0.1
  5. Brightness range 0.3 and 1.0
  6. Horizontal flip
  7. Vertical flip
* Reading the data
* Applying augmentation and then storing the data

## Loading Data
* Read the data
* Print the length of the data
* Print count plot for the classes
* Convert image to gray scale
* Splitting the dataset
* Convert the train and test to numpy array
* Plotting some of the images with lable
* Reshape the train and test and then normalize the dataset
* Printing the shape

## Model
* Making a CNN model
  1. Convolutional layer with 32 neurons and 5,5 filter
  2. Leaky relu as activation
  3. Maxpooling with filter 2,2
  4. Convolutional layer with 128 neurons and 5,5 filter
  5. Leaky relu as activation
  6. Convolutional layer with 64 neurons and 5,5 filter
  7. Leaky relu as activation
  8. Convolutional layer with 32 neurons and 5,5 filter
  9. Leaky relu as activation
  10. Maxpooling with filter 2,2
  11. Flatten Layer
  12. Dense layer with 1000 neurons
  13. Leaky relu as activation
  14. Dropout of 0.5
  15. Dense layer with 500 neurons
  16. Leaky relu as activation
  17. Dropout of 0.5
  18. Dense layer with 250 neurons
  19. Leaky relu as activation
  20. Dense layer with 1 neuron and sigmoid activation
* Print model summary
* Compile model
* Fit model for 100 epoch
* Get the testing accuracy and loss
* Plot the training and validation accuracy and loss
* Print classification report
* Print confusion matric
* Plot the incorrect and correct predictions with correct labels
* Send positive images to matlab

## Edge Detection
* Preprocess the input image
* Apply thresholding
* Apply morphological operations
* Display image with red border around the tumor
