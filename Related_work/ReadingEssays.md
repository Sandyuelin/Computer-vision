## Fundamentals and Basics

### Understanding LSTM Networks
- LSTMs are capable of keeping long and short memeory
- algorithm:
  - forget gate layer: $$f_t = \sigma (W_f \cdot [h_{t-1} , x_t]+ b_f)$$ which outputs numbers between 1(completely keep this) and 0(completely get rid o this) 
  - input gate layer: $$i_t = \sigma (W_i \cdot [h_{t-1} , x_t]+ b_i)$$ 
  - cell state update: $$\tilde{C}t = \tanh (W_C \cdot [h{t-1} , x_t]+ b_C)$$ which calculates the new candidate cell state
  - cell state: $$C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t$$ which updates the cell state
  - hidden state: $$o_t = \sigma (W_o \cdot [h_{t-1} , x_t]+ b_o)$$
     $$h_t = o_t \cdot \tanh(C_t)$$ which calculates the new hidden state
<img src="https://github.com/Sandyuelin/Computer-vision/blob/4499b23094026d20ddaa1243fffe08710902e1f4/Related_work/Screenshot%202024-06-11%20183847.png" alt="Screenshot" width="400"/>

- **The Unreasonable Effectiveness of RNNs** - Provides a broad perspective on the effectiveness of RNNs.
- **Recurrent Neural Network Regularization** - Focuses on techniques to regularize RNNs and improve performance.

## Convolutional Neural Networks (CNNs) for Image Recognition
- **ImageNet Classification with Deep CNNs** - Learn about the breakthroughs in image classification using deep convolutional networks.
### Deep Residual Learning for Image Recognition
- a stack of manuy residual blocks: add identical functions after Conv+ReLu+Conv
<img src="https://github.com/Sandyuelin/Computer-vision/blob/ad4a67f699727f5b1722349a5d0ecdbe5a877bd1/Related_work/Screenshot%202024-06-13%20190303.png" alt="Screenshot" width="400"/>

- basic block: two Conv 3 X 3
- bottleneck block --> more layers, less computations
 
- **Identity Mappings in Deep Residual Networks** - Explores improvements in ResNet architectures.

## Attention Mechanisms and Transformers
- **The Annotated Transformer** - An introduction to the Transformer architecture.
- **Attention Is All You Need** - Detailed insights into the Transformer model, crucial for understanding modern deep learning architectures.
- **Pointer Networks** - Learn about models that can handle complex structured data using attention mechanisms.

## Sequence Models and Translation
- **Neural Machine Translation by Jointly Learning to Align and Translate** - Provides insights into sequence-to-sequence models for translation.
- **Order Matters: Sequence to sequence for sets** - Understand the importance of order in sequences, relevant for many tasks including vision.

## Advanced Convolutional Networks and Context Aggregation
- **Multi-Scale Context Aggregation by Dilated Convolutions** - Study about dilated convolutions, useful in capturing multi-scale context in images.
- **GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism** - Learn how large models are trained efficiently, crucial as you progress to advanced topics.

## Generative Models and Autoencoders
- **Variational Lossy Autoencoder** - A type of generative model useful for unsupervised learning.

## Relational and Complex Reasoning
- **A Simple NN Module for Relational Reasoning** - Learn about relational networks important for tasks requiring complex reasoning.
- **Relational RNNs** - Dive into relational networks within the context of RNNs.

## Specialized Applications and Advanced Topics
- **Neural Turing Machines** - Explore models that can read and write to external memory.
- **Deep Speech 2: End-to-End Speech Recognition in English and Mandarin** - Understand end-to-end speech recognition, applicable principles to computer vision.
- **Neural Quantum Chemistry** - Explore an application of deep learning in quantum chemistry.
- **Quantifying the Rise and Fall of Complexity in Closed Systems** - Study theoretical aspects related to complexity in systems.

## Theory and Model Complexity
- **Keeping Neural Networks Simple by Minimizing the Description Length** - Study about model simplicity and the principle of Occam’s Razor in deep learning.
- **Scaling Laws for Neural LMs** - Learn how model performance scales with size and data.
- **A Tutorial Introduction to the Minimum Description Length Principle** - Gain a theoretical foundation in model selection and complexity.
- **PAGE 434 onwards: Komogrov Complexity** - Study more on complexity from an information theory perspective.
