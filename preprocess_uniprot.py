#!/usr/bin/env python3
"""
Preprocess UniProt data for XLSTM classification training
"""

import pandas as pd
import os

def preprocess_uniprot_data():
    """Convert raw UniProt data to format expected by XLSTM training"""
    
    # Read the raw data
    raw_file = "data/raw_data/uniprot/raw_data.csv"
    output_file = "data/uniprot/data.csv"
    
    print(f"Reading raw data from: {raw_file}")
    
    # Read the tab-separated file
    df = pd.read_csv(raw_file, sep='\t')
    
    print(f"Raw data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Process data for classification
    processed_data = []
    
    for idx, row in df.iterrows():
        if pd.notna(row['Gene Ontology IDs']) and pd.notna(row['Sequence']):
            # Split multiple GO terms (semicolon-separated)
            go_terms = str(row['Gene Ontology IDs']).split(';')
            
            for go_term in go_terms:
                go_term = go_term.strip()
                if go_term:  # Only add non-empty GO terms
                    processed_data.append({
                        'Sequence': row['Sequence'],
                        'Gene ontology IDs': go_term
                    })
    
    # Create DataFrame
    processed_df = pd.DataFrame(processed_data)
    
    print(f"Processed data shape: {processed_df.shape}")
    print(f"Unique GO terms: {processed_df['Gene ontology IDs'].nunique()}")
    print(f"Sample GO terms: {processed_df['Gene ontology IDs'].value_counts().head()}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save processed data
    processed_df.to_csv(output_file, index=False)
    print(f"Saved processed data to: {output_file}")
    
    return processed_df

if __name__ == "__main__":
    preprocess_uniprot_data()
