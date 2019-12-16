# MNIST-Digit-Recognition

## Binary Classifier
To classify [MNIST dataset](http://yann.lecun.com/exdb/mnist/) images of 7 and 8. <br />
To solve the convex optimization problem of SVM CVXOPT was used. One more SVM library called LibSVM was used to generate a new model. The results of both were compared and different kernels were also experimented with.

## Multi-class Classifier
To classify all the images of the dataset into 0-9 categories. <br />
Used LibSVM to generate the model and derive the support vectors. Validation set technique was used to derive the correct value of the parameter of Gaussian kernel. The accuracy was 97.23%.
