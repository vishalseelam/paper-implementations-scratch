import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
import random
from transformer import Transformer, create_transformer_model
from training_utils import (
    TransformerTrainer, 
    create_data_loaders, 
    count_parameters,
    plot_training_history
)

def generate_dummy_data(num_samples: int = 1000, src_vocab_size: int = 1000, 
                       tgt_vocab_size: int = 1000, max_len: int = 50) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Generate dummy data for demonstration purposes.
    This creates random sequences that simulate a translation task.
    """
    src_data = []
    tgt_data = []
    
    for _ in range(num_samples):
        # Generate source sequence
        src_len = random.randint(5, max_len)
        src_seq = [random.randint(1, src_vocab_size-1) for _ in range(src_len)]
        src_seq = [1] + src_seq + [2]  # Add start and end tokens
        
        # Generate target sequence (slightly different length)
        tgt_len = random.randint(5, max_len)
        tgt_seq = [random.randint(1, tgt_vocab_size-1) for _ in range(tgt_len)]
        tgt_seq = [1] + tgt_seq + [2]  # Add start and end tokens
        
        src_data.append(src_seq)
        tgt_data.append(tgt_seq)
    
    return src_data, tgt_data


def create_simple_vocabulary():
    """
    Create a simple vocabulary for demonstration.
    """
    vocab = {
        '<pad>': 0,
        '<sos>': 1,
        '<eos>': 2,
        '<unk>': 3,
    }
    
    # Add some common words
    words = ['hello', 'world', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 
             'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'these', 'those',
             'cat', 'dog', 'house', 'car', 'book', 'computer', 'phone', 'table', 
             'chair', 'door', 'window', 'tree', 'flower', 'water', 'food', 'time',
             'good', 'bad', 'big', 'small', 'happy', 'sad', 'new', 'old', 'fast', 'slow']
    
    for i, word in enumerate(words):
        vocab[word] = i + 4
    
    return vocab


def text_to_sequences(texts: List[str], vocab: dict, max_len: int = 50) -> List[List[int]]:
    """
    Convert text to sequences of token IDs.
    """
    sequences = []
    for text in texts:
        words = text.lower().split()
        sequence = [vocab.get('<sos>', 1)]
        for word in words:
            sequence.append(vocab.get(word, vocab.get('<unk>', 3)))
        sequence.append(vocab.get('<eos>', 2))
        
        # Pad or truncate to max_len
        if len(sequence) > max_len:
            sequence = sequence[:max_len]
        
        sequences.append(sequence)
    
    return sequences


def simple_translation_example():
    """
    Simple translation example with dummy data.
    """
    print("=" * 60)
    print("Simple Translation Example with Dummy Data")
    print("=" * 60)
    
    # Parameters
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    
    # Create model
    model = create_transformer_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_ff,
        dropout=0.1
    )
    
    print(f"Model created with {count_parameters(model):,} parameters")
    
    # Generate dummy data
    print("Generating dummy data...")
    src_data, tgt_data = generate_dummy_data(num_samples=1000, 
                                           src_vocab_size=src_vocab_size,
                                           tgt_vocab_size=tgt_vocab_size)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        src_data, tgt_data, batch_size=32, val_split=0.2
    )
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create trainer
    trainer = TransformerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        device=device
    )
    
    print("Starting training...")
    
    # Train for a few epochs
    history = trainer.train(num_epochs=3, save_path="transformer_model")
    
    # Plot training history
    plot_training_history(history, save_path="training_history.png")
    
    print("Training completed!")


def inference_example():
    """
    Demonstrate inference with the trained model.
    """
    print("\n" + "=" * 60)
    print("Inference Example")
    print("=" * 60)
    
    # Load model parameters (you would load from saved checkpoint in real scenario)
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    
    model = create_transformer_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Example inference function
    def translate(src_sequence: List[int], model: nn.Module, max_len: int = 50) -> List[int]:
        """
        Simple greedy translation function.
        """
        model.eval()
        with torch.no_grad():
            # Prepare source
            src = torch.tensor(src_sequence).unsqueeze(0).to(device)
            
            # Create source mask
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
            
            # Start with SOS token
            tgt_tokens = [1]  # SOS token
            
            for _ in range(max_len):
                tgt = torch.tensor(tgt_tokens).unsqueeze(0).to(device)
                
                # Create target mask
                tgt_len = tgt.size(1)
                tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1)
                tgt_mask = (tgt_mask == 0).unsqueeze(0).unsqueeze(0).to(device)
                
                # Forward pass
                output = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
                
                # Get next token
                next_token_logits = output[0, -1, :]
                next_token = torch.argmax(next_token_logits).item()
                
                tgt_tokens.append(next_token)
                
                # Stop if EOS token
                if next_token == 2:  # EOS token
                    break
            
            return tgt_tokens
    
    # Example translation
    src_sequence = [1, 45, 123, 456, 789, 2]  # Dummy source sequence
    translated = translate(src_sequence, model)
    
    print(f"Source sequence: {src_sequence}")
    print(f"Translated sequence: {translated}")


def architecture_analysis():
    """
    Analyze the architecture of the Transformer model.
    """
    print("\n" + "=" * 60)
    print("Transformer Architecture Analysis")
    print("=" * 60)
    
    # Create a model with standard parameters
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
    
    print(f"Model Parameters: {count_parameters(model):,}")
    print(f"Model Size: {count_parameters(model) * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Print model architecture
    print("\nModel Architecture:")
    print(model)
    
    # Test forward pass with dummy data
    print("\nTesting forward pass...")
    src = torch.randint(1, 1000, (2, 10))  # batch_size=2, seq_len=10
    tgt = torch.randint(1, 1000, (2, 8))   # batch_size=2, seq_len=8
    
    model.eval()
    with torch.no_grad():
        output = model(src, tgt)
        print(f"Input shape - Source: {src.shape}, Target: {tgt.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output vocabulary distribution shape: {output.shape}")


def main():
    """
    Main function to run all examples.
    """
    print("Transformer Model Implementation")
    print("Based on 'Attention Is All You Need' paper")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Run architecture analysis
    architecture_analysis()
    
    # Run simple translation example
    simple_translation_example()
    
    # Run inference example
    inference_example()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main() 