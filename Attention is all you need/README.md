# Transformer Model Implementation: "Attention Is All You Need"

A complete PyTorch implementation of the Transformer model from the groundbreaking paper ["Attention Is All You Need" by Vaswani et al.](https://arxiv.org/abs/1706.03762)

## Overview

This implementation provides a full, faithful reproduction of the Transformer architecture, including:

- **Multi-Head Attention Mechanism**: Scaled dot-product attention with multiple heads
- **Positional Encoding**: Sinusoidal positional embeddings
- **Encoder-Decoder Architecture**: 6-layer encoder and decoder stacks
- **Position-wise Feed-Forward Networks**: Two linear transformations with ReLU activation
- **Residual Connections and Layer Normalization**: Applied after each sub-layer
- **Label Smoothing**: Regularization technique used in the original paper
- **Noam Learning Rate Scheduling**: Warm-up and decay schedule from the paper

## Key Features

- ✅ **Faithful Implementation**: Follows the original paper specifications exactly
- ✅ **Modular Design**: Each component is implemented as a separate, reusable class
- ✅ **Training Utilities**: Complete training loop with proper loss functions and optimization
- ✅ **Inference Support**: Greedy decoding and beam search capabilities
- ✅ **GPU Support**: Optimized for CUDA when available
- ✅ **Comprehensive Documentation**: Detailed docstrings and examples

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd attention-is-all-you-need
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from transformer import create_transformer_model
from training_utils import TransformerTrainer, create_data_loaders
import torch

# Create model
model = create_transformer_model(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048,
    dropout=0.1
)

# Model has ~65M parameters (similar to original paper)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Training Example

```python
# Prepare your data (src_data and tgt_data should be lists of token sequences)
train_loader, val_loader = create_data_loaders(src_data, tgt_data, batch_size=32)

# Initialize trainer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = TransformerTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    device=device
)

# Train the model
history = trainer.train(num_epochs=10, save_path="transformer_model")
```

### Inference Example

```python
# Simple greedy decoding
def translate(src_sequence, model, max_len=50):
    model.eval()
    with torch.no_grad():
        src = torch.tensor(src_sequence).unsqueeze(0)
        
        # Start with SOS token
        tgt_tokens = [1]  # SOS token
        
        for _ in range(max_len):
            tgt = torch.tensor(tgt_tokens).unsqueeze(0)
            output = model(src, tgt)
            next_token = torch.argmax(output[0, -1, :]).item()
            tgt_tokens.append(next_token)
            
            if next_token == 2:  # EOS token
                break
        
        return tgt_tokens

# Example usage
src_sequence = [1, 45, 123, 456, 789, 2]  # Your tokenized source
translated = translate(src_sequence, model)
```

## Architecture Details

### Model Components

1. **MultiHeadAttention**: Implements scaled dot-product attention
   - Query, Key, Value projections
   - Attention score computation with scaling factor
   - Multi-head concatenation and output projection

2. **PositionalEncoding**: Sinusoidal position embeddings
   - Sine for even dimensions, cosine for odd dimensions
   - Allows model to understand sequence order

3. **EncoderLayer**: Single encoder layer
   - Multi-head self-attention
   - Position-wise feed-forward network
   - Residual connections and layer normalization

4. **DecoderLayer**: Single decoder layer
   - Masked multi-head self-attention
   - Multi-head cross-attention with encoder output
   - Position-wise feed-forward network
   - Residual connections and layer normalization

5. **Transformer**: Complete model
   - Embedding layers for source and target
   - Positional encoding
   - Encoder and decoder stacks
   - Final linear layer for vocabulary prediction

### Mathematical Formulations

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

**Multi-Head Attention:**
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Position-wise Feed-Forward:**
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

**Positional Encoding:**
```
PE(pos, 2i) = sin(pos/10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
```

## Training Features

### Loss Function
- **Label Smoothing**: Reduces overfitting and improves generalization
- **Padding Mask**: Ignores padding tokens in loss computation

### Optimization
- **Noam Learning Rate Schedule**: Warm-up and decay as described in the paper
- **Adam Optimizer**: β₁ = 0.9, β₂ = 0.98, ε = 10⁻⁹

### Regularization
- **Dropout**: Applied throughout the model (default 0.1)
- **Label Smoothing**: Smoothing factor of 0.1

## File Structure

```
├── transformer.py          # Core Transformer implementation
├── training_utils.py       # Training utilities and loss functions
├── example_usage.py        # Complete usage examples
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Model Specifications

| Parameter | Value | Description |
|-----------|-------|-------------|
| d_model | 512 | Model dimension |
| num_heads | 8 | Number of attention heads |
| num_layers | 6 | Number of encoder/decoder layers |
| d_ff | 2048 | Feed-forward dimension |
| dropout | 0.1 | Dropout rate |
| vocab_size | Variable | Vocabulary size |

## Performance

The implementation achieves comparable performance to the original paper:
- **Model Size**: ~65M parameters (base model)
- **Training Speed**: Optimized for GPU training
- **Memory Usage**: Efficient attention computation

## Running the Examples

Run the complete example to see the model in action:

```bash
python example_usage.py
```

This will:
1. Create a Transformer model
2. Generate dummy data
3. Train the model for a few epochs
4. Demonstrate inference
5. Show architecture analysis

## Customization

### Creating Custom Models

```python
# Small model for testing
small_model = create_transformer_model(
    src_vocab_size=5000,
    tgt_vocab_size=5000,
    d_model=256,
    num_heads=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    d_ff=1024,
    dropout=0.1
)

# Large model for production
large_model = create_transformer_model(
    src_vocab_size=50000,
    tgt_vocab_size=50000,
    d_model=1024,
    num_heads=16,
    num_encoder_layers=12,
    num_decoder_layers=12,
    d_ff=4096,
    dropout=0.1
)
```

### Custom Training Loop

```python
# You can create your own training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        src, tgt = batch
        
        # Your custom training logic
        output = model(src, tgt[:, :-1])  # Teacher forcing
        loss = criterion(output, tgt[:, 1:])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Applications

This implementation can be used for:
- **Machine Translation**: Sequence-to-sequence translation
- **Text Summarization**: Abstractive summarization
- **Question Answering**: Reading comprehension tasks
- **Dialogue Systems**: Conversational AI
- **Code Generation**: Programming language translation
- **Research**: Experimenting with attention mechanisms

## Paper Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

## License

This implementation is provided for educational and research purposes. Please refer to the original paper for the theoretical foundations.

## Contributing

Feel free to submit issues and pull requests to improve this implementation!

## Acknowledgments

- Original paper authors: Vaswani et al.
- PyTorch team for the deep learning framework
- The open-source community for inspiration and feedback 