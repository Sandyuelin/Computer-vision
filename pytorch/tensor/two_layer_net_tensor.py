import torch

"""
pytorch tensor is basically the same as a numpy array
same: n-dimensional array
difference:tensor can run on CPU/GPU
"""
device = torch.device('cuda')
N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N,D_in, device=device)
y = torch.randn(N,D_out, device=device)

w1 = torch.randn(D_in, H, device=device)
w2 = torch.randn(H, D_out, device=device)

lr = 1e-6
for t in range(500):
    h = x.mm(w1) # matrix multiplication
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= lr * grad_w1
    w2 -= lr * grad_w2




