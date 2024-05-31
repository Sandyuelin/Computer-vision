
# convolutional neural networks for computer vision
## reminder 
- write from scratch, debug, and train CNNs through python
- state of the art software tools such as Caffe, TensorFlow, and PyTorch: very modern and using the most recent ideas and methods
- what i need to use:
  python, linear algebra, college calculus, machine learning(cost functions, taking derivatives and performing optimization with gradient descent)
- challenges of recognition
   - illumination
   - deformation
   - occlusion
   - clutter
   - intraclass variation
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

## Loss function
- Multiclass SVM loss: 
$$\text{Given an example } (x_i,y_i) \text{ where } x_i\text{ is the image and where } y_i \text{is the label, and using shourthand for the scores vector,}$$ 
$$s_j(\mathbf{x_i}) = \mathbf{w_j} \cdot \mathbf{x_i}$$

$$L_i = \sum_{j \neq y_i} \max(0, s_j(\mathbf{x_i}) - s_{y_i}(\mathbf{x}_i) + \Delta) \,L \in (0, \infty)$$
 $$\text{delta here is a safe margin, loss over full dataset is average:}$$ 

$$L = \frac{1}{N} \sum_{i=1}^N L_i$$

the threshold at zero max(0, -) function is hinge loss, sometimes people using the squared hinge loss SVM (or L2-SVM) which uses the form max(0, -)^2 that penalizes violated margins quadratically instead of linearly
```
The loss function quantifies our unhappiness w/ predictions on the training set
```

## Regularization
extending the loss function w/ a **regularization penalty R(W)** that tells model should be simpler
- L2 regularization (shown below)
- L1 regularization
- elastic net (L1+L2)
- to be continued
$$R(W) = \sum_{k}\sum_{l}W_{k,l}^2$$

```
All we have to do now is to come up w/ a way to find the weights that minimize the loss
```


## Softmax classifier (Multiomial logistic regression)
$$L_i = - log(\frac{e^{s_{y_i}}}{\sum_j e^{s_j} })$$
steps: 
- exponentialize the scores
- normalize probability
- use logarithm

