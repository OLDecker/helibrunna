#!/usr/bin/env python3

import h5py
import sys

def test_h5_loading(h5_path, max_sequences=10):
    """Test H5 file loading with proper structure handling."""
    print(f"Testing H5 file: {h5_path}")
    
    sequences = []
    with h5py.File(h5_path, 'r') as f:
        print(f"H5 file keys: {list(f.keys())}")
        
        # Get first dataset for testing
        raw_data_keys = [key for key in f.keys() if key.startswith('raw_data_')]
        print(f"Found raw_data datasets: {raw_data_keys[:3]}...")
        
        dataset_name = raw_data_keys[0]  # Test with first dataset
        print(f"Testing with dataset: {dataset_name}")
        dataset = f[dataset_name]
        
        print(f"Dataset {dataset_name} shape: {dataset.shape}")
        print(f"Dataset {dataset_name} dtype: {dataset.dtype}")
        
        for i in range(min(max_sequences, len(dataset))):
            entry = dataset[i]
            
            # Handle numpy structured array entries (protein_id, sequence)
            if hasattr(entry, 'item') and isinstance(entry.item(), tuple):
                # This is a numpy void object containing (protein_id, sequence)
                protein_id, sequence = entry.item()
                
                # Decode bytes to string if necessary
                if isinstance(sequence, bytes):
                    sequence = sequence.decode('utf-8')
                if isinstance(protein_id, bytes):
                    protein_id = protein_id.decode('utf-8')
                    
                sequences.append(sequence)
                print(f"Entry {i}: ID={protein_id}, Seq={sequence}")
            else:
                print(f"Entry {i}: Unexpected format: {type(entry)}")
    
    print(f"\nSuccessfully loaded {len(sequences)} sequences")
    if sequences:
        print(f"Sample sequence: {sequences[0]}")
    
    return sequences

if __name__ == "__main__":
    h5_path = "/gpfs/bwfor/work/ws/hd_il278-protein_go_term_prediction/protein_go_term_prediction/data/data/uniparc/uniparc_train_sorted.h5"
    sequences = test_h5_loading(h5_path, max_sequences=5)
