#!/usr/bin/env python3
"""
Simple test script to verify the Transformer implementation works correctly.
"""

import torch
import numpy as np
from transformer import create_transformer_model, PositionalEncoding, MultiHeadAttention
from training_utils import count_parameters, LabelSmoothingLoss, NoamOptimizer

def test_positional_encoding():
    """Test positional encoding implementation."""
    print("Testing Positional Encoding...")
    
    pe = PositionalEncoding(d_model=512, max_len=100)
    x = torch.randn(50, 2, 512)  # seq_len=50, batch_size=2, d_model=512
    
    output = pe(x)
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    print("✓ Positional encoding test passed")

def test_multi_head_attention():
    """Test multi-head attention implementation."""
    print("Testing Multi-Head Attention...")
    
    mha = MultiHeadAttention(d_model=512, num_heads=8, dropout=0.1)
    batch_size, seq_len, d_model = 2, 10, 512
    
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    output = mha(query, key, value)
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("✓ Multi-head attention test passed")

def test_transformer_model():
    """Test the complete Transformer model."""
    print("Testing Transformer Model...")
    
    # Create model
    model = create_transformer_model(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1
    )
    
    # Test forward pass
    batch_size, src_len, tgt_len = 2, 10, 8
    src = torch.randint(1, 1000, (batch_size, src_len))
    tgt = torch.randint(1, 1000, (batch_size, tgt_len))
    
    output = model(src, tgt)
    expected_shape = (batch_size, tgt_len, 1000)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    # Check parameter count
    param_count = count_parameters(model)
    print(f"✓ Model parameters: {param_count:,}")
    print("✓ Transformer model test passed")

def test_label_smoothing_loss():
    """Test label smoothing loss implementation."""
    print("Testing Label Smoothing Loss...")
    
    vocab_size = 1000
    batch_size, seq_len = 2, 10
    
    criterion = LabelSmoothingLoss(vocab_size=vocab_size, padding_idx=0, smoothing=0.1)
    
    pred = torch.randn(batch_size, seq_len, vocab_size)
    target = torch.randint(1, vocab_size, (batch_size, seq_len))
    
    loss = criterion(pred, target)
    assert loss.item() > 0, "Loss should be positive"
    print("✓ Label smoothing loss test passed")

def test_noam_optimizer():
    """Test Noam optimizer implementation."""
    print("Testing Noam Optimizer...")
    
    model = create_transformer_model(src_vocab_size=100, tgt_vocab_size=100, d_model=128)
    
    optimizer = NoamOptimizer(
        model_size=128,
        factor=2.0,
        warmup=4000,
        optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    )
    
    # Test learning rate calculation
    rate_1 = optimizer.rate(1)
    rate_100 = optimizer.rate(100)
    rate_1000 = optimizer.rate(1000)
    rate_5000 = optimizer.rate(5000)
    rate_10000 = optimizer.rate(10000)
    
    print(f"  Learning rates: step 1={rate_1:.6f}, step 100={rate_100:.6f}, step 1000={rate_1000:.6f}, step 5000={rate_5000:.6f}, step 10000={rate_10000:.6f}")
    
    # Rate should increase initially, then decrease after warmup (4000 steps)
    assert rate_100 > rate_1, "Learning rate should increase during warmup"
    assert rate_1000 > rate_100, "Learning rate should continue increasing during warmup"
    assert rate_10000 < rate_5000, "Learning rate should decrease after warmup"
    print("✓ Noam optimizer test passed")

def test_attention_mask():
    """Test attention mask functionality."""
    print("Testing Attention Masks...")
    
    model = create_transformer_model(src_vocab_size=100, tgt_vocab_size=100, d_model=128)
    
    # Create sequences with padding
    src = torch.tensor([[1, 2, 3, 4, 0, 0], [1, 2, 0, 0, 0, 0]])  # batch_size=2, seq_len=6
    tgt = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, 4, 0]])  # batch_size=2, seq_len=5
    
    # Test that model can handle padded sequences
    output = model(src, tgt)
    expected_shape = (2, 5, 100)  # batch_size, tgt_len, vocab_size
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    print("✓ Attention mask test passed")

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Transformer Implementation Tests")
    print("=" * 60)
    
    try:
        test_positional_encoding()
        test_multi_head_attention()
        test_transformer_model()
        test_label_smoothing_loss()
        test_noam_optimizer()
        test_attention_mask()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! Implementation is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    run_all_tests() 