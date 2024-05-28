
# convolutional neural networks for computer vision
## reminder 
- write from scratch, debug, and train CNNs through python
- state of the art software tools such as Caffe, TensorFlow, and PyTorch: very modern and using the most recent ideas and methods
- what i need to use:
  python, linear algebra, college calculus, machine learning(cost functions, taking derivatives and performing optimization with gradient descent)
## K-Nearest Neighbors 
- image classification 
    - **training set** of images and labels
    - predict labels on the **test set**
- K-Nearest Neighbors classifier predicts based on nearest training examples
- Distance metric and K are hyperparameters
    - L1 (Manhattan) distance
    - L2 (Euclidean) distance
  
$$d_{\text{Manhattan}}(\mathbf{I_1}, \mathbf{I_2}) = \sum_{i=p} |I_i^p - I_2^p|$$

$$d_{\text{Euclidean}}(\mathbf{I_1},\mathbf{I_2}) = \sqrt{\sum_{i=p} (I_1^p - I_2^p)^2}$$
- Choose hyperparameters using validation set
    - hyperparameters are the choices about the algorithm thant we set rather than learn
    - problem-dependent
    -  only run the test set once at the vert end (to be honest w the result)

## Linear classification
parametric approach
$$f(x,W) = Wx +b$$
- linear classifier is the inner product of weights or parameters matrix and pixel column plus bias
- row of weight matrix represents class
