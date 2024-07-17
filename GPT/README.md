### backpropagation
- compute gradients __with respect to loss__
- use chain rule 
```
in matrix calculus, C = A.dot B
the gradient of C with respect to A is given transpose of B
```
#### initialization + forward
```python
from torch import functional as F
def __init__(self, vocab_size): 
    # is a constructor and
    # is automatically invoked when a new instance of class is created
    super().__init__()
    # each token directly reads off the logits for the next token
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

def forward(self, idx, targets= None):
    logits = self.token_embedding_table(idx) 
    # it returns out a (B, T, C) batch size by time by channel
    # look at the documentation of pytorch
    # it expects (minibatch, C) in cross entropy loss
    # so we need to reshape the logits and targets
    if targets is None:
        loss = None
    else:    
        B, T, C = logits.shape
        logits = logits.view(B*T,C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
    return logits, loss
```
- tokenize: characters seperate into token,
represented as integer 
- encoder: take a string, output a list of integers
- decoder: take a list of integers, output a string
- batch size: how many independent sequences will be processed in parallel
- block size: maximum context length for predictions
## self-attention
- every token will emit two vectors: query and key
  - query: what do i look for
  - key: what do i contain
  - the dot product of query and key is weight
```python
x = torch.randn(B,T,C)
head_size = 16
key = nn.Linear(C, head_size, bias = False) # input size, output size
query = nn.Linear(C, head_size, bias = False)
value = nn.Linear(C, head_size, bias = False)
k = key(x) # (B,T,16)
q = query(x) # (B,T,16)
wei = q @ k.transpose(-2,-1) # (B,T,16) @ (B,16,T)---> (B,T,T) 
tril = torch.tril(torch.ones(T,T))
wei = wei.masked_fill(tril==0, float('-inf'))
wei = F.softmax(wei, dim=-1)
v = value(x)
out = wei @ v
```
note: 
- nodes in a directed graph aggregating info w a weighted sum from all nodes that point to them,
with data-dependent weights
- self in self attention because key, query and value come from x; cross attention may have other external sources of conditions or contexts on key and query
