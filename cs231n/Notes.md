
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

```
Image Features:
- Color Histogram
-  Histogram of Oriented Gradients(HoG)
- Bag of Words (create a codebook and envode images)****
```
## Loss function
### Data loss/ Hinge loss
$$\text{Given an example } (x_i,y_i) \text{ where } x_i\text{ is the image and where } y_i \text{is the label, and using shourthand for the scores vector,}$$ 
$$s_j(\mathbf{x_i}) = \mathbf{w_j} \cdot \mathbf{x_i}$$

$$L_i = \sum_{j \neq y_i} \max(0, s_j(\mathbf{x_i}) - s_{y_i}(\mathbf{x_i}) + \Delta)  \;L \in (0, \infty)$$
 $$\text{delta here is a safe margin, loss over full dataset is average:}$$ 

$$L = \frac{1}{N} \sum_{i=1}^N L_i$$

the threshold at zero max(0, -) function is hinge loss, sometimes people using the squared hinge loss SVM (or L2-SVM) which uses the form max(0, -)^2 that penalizes violated margins quadratically instead of linearly
```
The loss function quantifies our unhappiness w/ predictions on the training set
```


### Regularization loss
extending the loss function w/ a **regularization penalty R(W)** that tells model should be simpler
- L2 regularization (shown below)
- L1 regularization
- elastic net (L1+L2)
- to be continued
$$R(W) = \sum_{k}\sum_{l} W_{k,l}^2$$

### Support Vector Machine (SVM)
$$L = \frac{1}{N} \sum_{i=1}^N \sum_{j \neq y_i} \max(0, s_j(\mathbf{x_i}) - s_{y_i}(\mathbf{x_i}) + \Delta) + \lambda \sum_{k}\sum_{l} W_{k,l}^2$$

### Computing the gradient
The gradient of the data loss with respect to $W$ for a given training example $i$ is:

$$\frac{\partial L_i}{\partial W} = 
\begin{cases} 
-X_i & \text{if } j = y_i \text{ and } s_j - s_{y_i} + \Delta > 0 \\
X_i & \text{if } j \neq y_i \text{ and } s_j - s_{y_i} + \Delta > 0 \\
0 & \text{otherwise}
\end{cases}$$

The gradient of the regularization loss with respect to $W$ is:

$$\frac{\partial L_{\text{reg}}}{\partial W} = 2 \lambda W$$
```
All we have to do now is to come up w/ a way to find the weights that minimize the loss
```


## Softmax classifier (Multiomial logistic regression)
$$L_i = - log(\frac{e^{s_{y_i}}}{\sum_j e^{s_j} })$$
steps: 
- exponentialize the scores
- normalize probability
- use logarithm

## Optimization
Goal: find **W** to minimize the loss function
- numerical gradient
- analytic gradient
- gradient check: use numerical one to check analytic one


## Stochastic gradent descent (SGD)
$$\nabla_W L(W) = \frac{1}{N} \sum_{i=1}^N\nabla_W L_i(x_i,y_i,W)+\lambda \nabla_W R(W)$$

(nabla symbol here denotes gradient operator)
hyperparameters:
- weight initialization
- number of steps
- learning rate
- batch size
- data sampling

problems w SGD: 
- what if the loss function has a local minimum or saddle point
- gradients come from minibatches so they can be noisy
<br>

solutions: **SGD+Momentum**
$$v_{t+1} = \rho v_t +\nabla f(x_t)$$
$$x_{t+1} = x_t -\alpha v_{t+1}$$
```
v = 0
for t in range(num_steps):
  dw = compute_gradient(w)
  v = rho*v +dw
  w -= learning_rate*v 

```
rho is friction
<br>

**Adam: RMSProp + Momentum**
[Adam](https://arxiv.org/pdf/1412.6980)
```
moment1 = 0
moment2 = 0
for t in range(num_steps):
   dw = compute_gradient(w)
   moment1 = beta1*moment1+ (1-beta1)*dw
   moment2 = beta2*moment2 + (1-beta2)*dw*dw
   w -= learning_rate * moment1 / (moment2.sqrt() +1e-7) 
```
 Adam with beta1=0.9 beta2=0.999 and learning_rate = 1e-3,5e-4,1e-4 is great starting point for many models

## Neural Network


Multi-Layer Network(MLP) fully connects elements
- input layer
- hidden layer
- output layer
- ReLU (default choice) activation function
    $$ReLU(z) = max(0,z)$$ 
- Softmax loss function
<br>

universal approximation:
- build a bump function using hidden units
- use narrower gap between bump and increase bump numbers to increase the fidelity of representation 
<br>

convex functions
$$f(t x_1 +(1-t) x2) \leq t f(x_1) + (1-t) f(x_2)$$
- intuition: a convex function is a bowl
- easy to optimize since it has theoretical guarantess about converging to global minimum
<br>

**backpropagation**
- Forward pass: compute  outputs
- Backward pass: compute derivatives (use chain rule)
using chain rule
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial o} \frac{\partial o}{\partial w}$$
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial o} \frac{\partial o}{\partial h} \frac{\partial h}{\partial w}$$
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial o} \frac{\partial o}{\partial h} \frac{\partial h}{\partial o} \frac{\partial o}{\partial w}$$
downstream gradient = local gradient * upstream gradient

- patterns in gradient flow
    - add gate: gradient distributor (same)
    - copy gate: gradient adder 
    - mul gate: swap muliplier
    - max gate: gradient router (flow to the max )
- backprop with vectors
   - Jacobian vectors
- backprop with matrices/tensors
    - local Jacobian matrices
        - x:[NxD] w: [DxM] by matrix multiply y = xw y:[NxM]
        - dL/dx = (dL/dy) w^T
        - [NxD]   [NxM]   [MxD]
- back: higher-order derivatives: **Hessian matrix**
    
## Convolutional networks
*Fully-connected network flatten the image into vectors*
- components  
  - fully-connected layers
  - Conv + ReLU (Since W2W1 = linear classifier )
  - pooling layers: 4X4 -> 2X2
  - normalization
  ![stdDraw](https://github.com/Sandyuelin/Computer-vision/blob/807f5ae0d990842cd1031beb7630c7a145ca1e39/cs231n/Screenshot%202024-06-12%20151703.png)
 
**Convolutional filters**
<br>

- what do convolutional filters learn?
    - first-layer conv filters: local image templates (often learn oriented edges, opposing colors)
- Same padding: output and input have the same size when P = (K-1)/2
   - input: W
   - filter: K
   - padding: P
   - output: W-K+1+2P
- Strided convolution (Downsample)
   - input
   - filter
   - padding
   - stride: S
   - output: (W-K+2P)/S+1
<br>

Classic architecture: **[Conv, ReLU, Pool] x N, flatten, [FC,ReLU] X N, FC**


## Hardware and software

- CPU: central processing unit
- GPU: graphics processing unit

|           | Cores                | Clock Speed (GHz)   | Memory        | Price | TFLOP/sec  |
|-----------|----------------------|---------------------|---------------|-------|------------|
| **CPU**   |                      |                     |               |       |            |
| Ryzen 9   | 16                   | 3.5                 | System RAM    | $749  | ~4.8 FP32  |
| 3950X     | (32 threads with     | (4.7 boost)         |               |       |            |
|           | hyperthreading)      |                     |               |       |            |
| **GPU**   |                      |                     |               |       |            |
| NVIDIA    | 4608                 | 1.35                | 24 GB GDDR6   | $2499 | ~16.3 FP32 |
| Titan RTX |                      | (1.77 boost)        |               |       |            |

- **CPU**: Fewer cores, but each core is much faster and much more capable; great at sequential tasks.
- **GPU**: More cores, but each core is much slower and "dumber"; great for parallel tasks.

## Training neural networks

### activation functions
- signoid 
   - squashes numbers to range [0,1]
   - saturated neurons kill the gradients
   - signmoid outputs are not zero-centered
   - exp() a bit compute expensive (for CPU)
- tanh
   - squashes numbers to range[-1,1]
   - zero-centered
   - still kill gradients
- ReLu
   - cheap, simple, does not saturate
   - not zero-centered output: maybe less of a concern
- Leacky ReLU $f(x)=max(0.01x,x)$
   - will not die
- ELU: exponential linear unit 
<br>

$f(x)=x$  if x > 0 $f(x) = \alpha (exp(x)-1)$ if x <= 0
- SELU: scaled exponential linear unit

<br> 

**dont think too hard, just use ReLU and dont use sigmoid or tanh**
<br>
whats going on: weight initialization(Xaview, Kaiming); regularization(dropout,mixup)

### learning rate decay
- step
- cosine
- linear
- inverse sqrt
- constant

## Recurrent neural networks (RNN)
- process sequences (image captioning, video classification)
- Vanilla recurrent neural networks
   - x -> RNN -> y
   - which consists a single hidden vector **h**
   $$h_t = f_W (h_{t-1} , x_t)$$
   $$h_t = tanh(W_{hh} h_{t-1} + W_{xh} x_t + bias)$$
   $$y_t = W_{hy} h_t$$
    <img src="https://github.com/Sandyuelin/Computer-vision/blob/90dd3f71d0ab3acf1a6de2103bf2354ab20fa0cc/cs231n/Screenshot%202024-06-17%20202249.png" alt="Screenshot" width="400"/>
     <img src="https://github.com/Sandyuelin/Computer-vision/blob/90dd3f71d0ab3acf1a6de2103bf2354ab20fa0cc/cs231n/Screenshot%202024-06-17%20202314.png" alt="Screenshot" width="400"/>
      <img src="https://github.com/Sandyuelin/Computer-vision/blob/90dd3f71d0ab3acf1a6de2103bf2354ab20fa0cc/cs231n/Screenshot%202024-06-17%20202439.png" alt="Screenshot" width="400"/>
- [LSTM explaination](https://github.com/Sandyuelin/Computer-vision/blob/b0d5c08877438f01991eb437884b92eb4dbe96cc/Related_work/ReadingEssays.md)

## Attention
#### attend
- image captioning w/ RNNs and Attention
   - each step of decoder use a different context vector that looks at different parts of the input image

#### Attention layer
- inputs:
  - Query vectors: Q ($N_Q \times D_Q$) context vector
  - Input vectors: X ($N_X \times D_X$)
  - Key matrix: $W_K$ ($D_X \times D_Q$) transform input vectors into key vectors, key vectors will be compared against query vectors to compute attention scores
  - Value matrix: $W_V$ ($D_X \times D_V$)
- computation
   - Key vectors: $K= XW_K$
   - Value vectors:$V = XW_V$
   - Similarity (Attention scores): $E = QK^T$  $E_{i,j} = Q_i \cdot K_j / \sqrt(D_Q)$
   - Attention weights: $A = softmax(E, dim =1)$
   - Output: $Y= AV$ $Y_i = \sum_j A_{i,j}V_j$
![StdDraw](https://github.com/Sandyuelin/Computer-vision/blob/df41b7c508209d23b6437b0c19b4b9d701135252/cs231n/screenshots/Attention_graph.png)

