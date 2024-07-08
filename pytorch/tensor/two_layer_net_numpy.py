import numpy as np

'''
a fully connected ReLU network with one hidden layer and no biases,
trained to predict y from x using Euclidean error
'''
N, D_in, H, D_out = 64, 1000, 100, 10
# batch size,input dimension,hidden dimension,output dimension
x = np.random.randn(N, D_in) # random input data
y = np.random.randn(N, D_out)
# .randn() creates an array of randon values w/ specified shape
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

lr = 1e-6
# forward pass: compute predicted y
for t in range(500):
  h = x.dot(w1)
  h_relu =np.maximum(h, 0)
  y_pred = h_relu.dot(w2)

  loss = np.square(y_pred - y).sum()
  print(t, loss)

  # backprop to compute gradients of w1 and w2 with respect to loss
  grad_y_pred = 2.0 * (y_pred - y) #
  grad_w2 = h_relu.T.dot(grad_y_pred)
  grad_h_relu = grad_y_pred.dot(w2.T)
  grad_h = grad_h_relu.copy()
  grad_h[h<0] = 0
  grad_w1 = x.T.dot(grad_h)

  # update weights
  w1 -= lr * grad_w1
  w2 -= lr * grad_w2
