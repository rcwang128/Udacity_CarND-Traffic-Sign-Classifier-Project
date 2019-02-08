# **CarND-term1 Traffic Sign Recognition** 

## Project 3

### Harry Wang

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Writeup / README that includes all the rubric points and how I addressed each one.

Here is a link to my [project code](https://github.com/rcwang128/Udacity_CarND-Traffic-Sign-Classifier-Project)



### Data Set Summary & Exploration

#### 1. Basic summary of the data set. 

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799.

* The size of the validation set is 4410.

* The size of test set is 12630.

* The shape of a traffic sign image is 32x32x3.

* The number of unique classes/labels in the data set is 43.

  ```python
  import numpy as np
  
  n_train = len(X_train)
  n_validation = len(X_valid)
  n_test = len(X_test)
  image_shape = X_train[0].shape
  
  n_classes = np.unique(np.concatenate((y_train, y_valid, y_test))).shape[0]
  
  print("Number of training examples =", n_train)
  print("Number of validation examples =", n_validation)
  print("Number of testing examples =", n_test)
  print()
  print("Image data shape =", image_shape)
  print("Number of classes =", n_classes)
  ```

  ```
  Number of training examples = 34799
  Number of validation examples = 4410
  Number of testing examples = 12630
  
  Image data shape = (32, 32, 3)
  Number of classes = 43
  ```

  

#### 2. An exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed with respect to classes.

![dataset.png](https://github.com/rcwang128/Udacity_CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/dataset.png?raw=true)



### Design and Test a Model Architecture

#### 1. Preprocessed image data. 

The image data is being normalized with`(pixel - 128)/ 128` function after converting to grayscale. A shape of `32x32x1` will be used as training input.

```python
# Normalize the iamges
def normalize(img):
    return (img - 128) / 128

X_train = normalize(np.sum(X_train/3, axis=3, keepdims=True))
X_valid = normalize(np.sum(X_valid/3, axis=3, keepdims=True))
X_test = normalize(np.sum(X_test/3, axis=3, keepdims=True))

print(X_train[index].shape)
```

```
(32, 32, 1)
```




#### 2. Description of Model Architecture.

A 5-layer LeNet architecture is being used here in this project:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 normalized image |
| Convolution layer 1 | 1x1 stride, valid padding, outputs 28x28x6 |
| Activation 1	| ReLU |
| Max pooling 1	     | 2x2 stride,  outputs 14x14x6 |
| Convolution layer 2	| 1x1 stride, valid padding, outputs 10x10x16 |
| Activation 2	| ReLU        					|
| Max pooling 2	| 2x2 stride,  outputs 5x5x16 |
| Flatten layer | outputs 400 |
| Fully connected layer 3 | outputs 120 |
| Activation 3 | ReLU |
| Fully connected layer 4 | outputs 84 |
| Activation 4 | ReLU |
| Dropout layer | Keep probability 0.75 |
| Fully connected out layer | outputs 43 |



#### 3. Description of model training procedures.

To train the model, I used below parameters.

```python
# Parameters
EPOCHS = 40
BATCH_SIZE = 128

# Network Parameters
n_classes = 43     # Total classes = 43
dropout = 0.75   # Probability to keep units
```

Initialization of weights and biases for each layer.

```python
# Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
mu = 0
sigma = 0.1

weights = {
    'wc1': tf.Variable(tf.truncated_normal(shape=(5,5,1,6), mean=mu, stddev=sigma)),
    'wc2': tf.Variable(tf.truncated_normal(shape=(5,5,6,16), mean=mu, stddev=sigma)),
    'wd1': tf.Variable(tf.truncated_normal(shape=(400,120), mean=mu, stddev=sigma)),
    'wd2': tf.Variable(tf.truncated_normal(shape=(120,84), mean=mu, stddev=sigma)),
    'out': tf.Variable(tf.truncated_normal(shape=(84,n_classes), mean=mu, stddev=sigma))}

biases = {
    'bc1': tf.Variable(tf.zeros([6])),
    'bc2': tf.Variable(tf.zeros([16])),
    'bd1': tf.Variable(tf.zeros([120])),
    'bd2': tf.Variable(tf.zeros([84])),
    'out': tf.Variable(tf.zeros([n_classes]))}
```

Below are the implementation of my 5-layer LeNet architecture, where `conv2d(x, W, b, strides=1)` is the convolution function with ReLU return while `maxpool2d(x, k=2)` is the max pooling function following the convolution layer. 

```python
def LeNet(x):
    # For feature maps outputting
    global conv1, conv2
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    # Valid padding: (inpur - filter + 1) / strides
    # Pooling: Input = 28x28x6. Output = 14x14x6.
    # Valid padding: (input - filter) / strides + 1
    conv1 = conv2d(x, weights['wc1'], biases['bc1']) # (32-5+1)/1 = 28
    conv1 = maxpool2d(conv1, k=2) # (28-2)/2+1 = 14
    
    # Layer 2: Convolutional. Output = 10x10x16.
    # Pooling: Input = 10x10x16. Output = 5x5x16.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2']) # (14-5+1)/1 = 10
    conv2 = maxpool2d(conv2, k=2) # (10-2)/2+1 = 5
    
    # Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    # Activation.
    fc1 = tf.matmul(fc0, weights['wd1']) + biases['bd1']
    fc1 = tf.nn.relu(fc1)
    #fc1 = tf.nn.dropout(fc1, dropout)
    
    # Layer 4: Fully Connected. Input = 120. Output = 84.
    # Activation.
    fc2 = tf.matmul(fc1, weights['wd2']) + biases['bd2']
    fc2 = tf.nn.relu(fc2)
    
    # Apply Dropout
    fc2 = tf.nn.dropout(fc2, dropout)

    # Layer 5: Fully Connected. Input = 84. Output = 43. Output layer.
    logits = tf.matmul(fc2, weights['out']) + biases['out']
    
    return logits
```

Adam optimizer is used with a learning rate of `0.001`. The training pipeline follows the procedure of minimizing loss by calculating the cross entropy. 

```python
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
```

The model is being trained with defined parameters.

```python
save_file = './lenet'
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, save_file)
    print("Model saved")
```



#### 4. Description of the approach taken for finding the solution.

A validation set accuracy of `0.93` is reached using this model.

```python
import tensorflow as tf

save_file = './lenet'
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, save_file)
            
    validation_accuracy = evaluate(X_valid, y_valid)
    print("Validation Accuracy = {:.3f}".format(validation_accuracy))
```

```
INFO:tensorflow:Restoring parameters from ./lenet
Validation Accuracy = 0.930
```



Test set accuracy of `0.922` is achieved.

```python
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
```

```
INFO:tensorflow:Restoring parameters from ./lenet
Test Accuracy = 0.922
```



**Discussion on parameter tuning**

An additional dropout layer is added in this LeNet architecture. Dropout is a simple but powerful regularization technique for neural networks. During training procedures, if neurons are randomly dropped out, the other ones will have to step in and handle the representation required to make predictions for the missing neurons. In this project, the keep probability is tuned as `0.75`.

Besides adding dropout layer, epochs and learning rate are the main parameters being tuned for this training model. More epochs simply means more iteration of the training cycles both forward and backward through entire neural network. The number of epochs have been tuned to avoid overfitting and underfitting. An epoch number of `40 ` is used for this training model.

Generally, high learning rate would result in fast learning at beginning but getting saturated quickly too without further improvement. Small learning rate is used for this training, which tends to find local minima more likely with the sacrifice of training time and epochs. A learning rate of `0.001` is used in this project.



### Test the Model on New Images

#### 1. Choose 5 German traffic signs found on the web and provide them in the report. 

Here are five German traffic signs that I found on the web:

![test_new_images.png](https://github.com/rcwang128/Udacity_CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/test_new_images.png?raw=true)

New images have been resized and normalized the same way as training set （32x32x1）before feeding into neural network.



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. 

Here are the results of the prediction of these 5 new images. 

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Road work    |               Road work               |
|             Priority road             | Priority road |
| Right-of-way at the next intersection	| Right-of-way at the next intersection	|
|           Children crossing           |             *Ahead only*              |
| Yield			| Yield      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The test sample size is small in this case.

```Python
# Correct image classes
y_test_new_img = [25, 12, 11, 28, 13]

with tf.Session() as sess:
    saver.restore(sess, save_file)
            
    test_img_accuracy = evaluate(X_test_new_img, y_test_new_img)
    print("Test Accuracy = {:.3f}".format(test_img_accuracy))
```

```
INFO:tensorflow:Restoring parameters from ./lenet
Test Accuracy = 0.800
```

Top 5 softmax probabilities are presented for each images using the `tf.nn.top_k` function.

![new_image_pred.png](https://github.com/rcwang128/Udacity_CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/new_image_pred.png?raw=true)

One of the major reason that test image 4 is mis-recognized as "Road work" may due to less number of training sets. "Children crossing" (class#28) has relatively small training samples comparing with others if looking through dataset bar plot. And "Children crossing" sign has quite a few variants and their features are similar as a lot other traffic signs. Those could also contribute to the prediction error.



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. 

First CNN layer has depth of 6 while 2nd CNN layer has depth of 16. Their feature maps are visualized as images shown below for a given test image.

![featuremap.png](https://github.com/rcwang128/Udacity_CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/featuremap.png?raw=true)
