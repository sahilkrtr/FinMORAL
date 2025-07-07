#!/usr/bin/env python3
"""
Step 2: Batch Candidate Generation
==================================
Process candidates in small batches to avoid API rate limits
"""

import json
import time
import random
from step2_candidate_generation import CandidateAnswerGenerator

def process_in_batches():
    """Process candidates in batches of 3 samples from each dataset"""
    print("Step 2: Batch Candidate Generation")
    print("=" * 50)
    
    # Load all examples
    with open('step1_processed_data.jsonl', 'r', encoding='utf-8') as f:
        all_examples = [json.loads(line) for line in f]
    
    print(f"Total examples available: {len(all_examples)}")
    
    # Separate examples by dataset
    dataset_examples = {}
    for example in all_examples:
        dataset = example['dataset']
        if dataset not in dataset_examples:
            dataset_examples[dataset] = []
        dataset_examples[dataset].append(example)
    
    print(f"Datasets found: {list(dataset_examples.keys())}")
    for dataset, examples in dataset_examples.items():
        print(f"   {dataset}: {len(examples)} examples")
    
    # Initialize generator
    generator = CandidateAnswerGenerator()
    
    # Process 30 samples from each dataset (10 batches of 3 samples each)
    all_candidates = []
    batch_size = 3
    num_batches = 10
    samples_per_dataset = batch_size * num_batches  # 30 samples per dataset
    
    print(f"\nProcessing {samples_per_dataset} samples from each dataset")
    print(f"   Total: {samples_per_dataset * len(dataset_examples)} samples")
    
    for dataset_name, dataset_examples_list in dataset_examples.items():
        print(f"\nPROCESSING DATASET: {dataset_name.upper()}")
        print("=" * 50)
        
        # Process this dataset in batches
        for batch_num in range(num_batches):
            print(f"\n{dataset_name.upper()} - BATCH {batch_num + 1}/{num_batches}")
            print("-" * 40)
            
            # Select samples for this batch
            start_idx = batch_num * batch_size
            end_idx = start_idx + batch_size
            batch_examples = dataset_examples_list[start_idx:end_idx]
            
            print(f"Processing examples {start_idx + 1}-{end_idx}")
            
            # Generate candidates for this batch
            batch_candidates = []
            for example in batch_examples:
                candidates = generator.generate_candidates_for_example(example)
                batch_candidates.extend(candidates)
                
                # Small delay between examples
                time.sleep(0.5)
            
            all_candidates.extend(batch_candidates)
            print(f"{dataset_name} Batch {batch_num + 1} completed: {len(batch_candidates)} candidates")
            
            # Wait between batches to avoid rate limits
            if batch_num < num_batches - 1:  # Don't wait after last batch
                wait_time = 10  # 10 seconds between batches
                print(f"Waiting {wait_time} seconds before next batch...")
                time.sleep(wait_time)
        
        print(f"{dataset_name.upper()} completed: {samples_per_dataset} samples")
        
        # Wait between datasets
        if dataset_name != list(dataset_examples.keys())[-1]:  # Not the last dataset
            wait_time = 15  # 15 seconds between datasets
            print(f"Waiting {wait_time} seconds before next dataset...")
            time.sleep(wait_time)
    
    # Save all candidates (append to existing file)
    output_file = "step2_candidates.jsonl"
    
    # Load existing candidates if file exists
    existing_candidates = []
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_candidates = [json.loads(line) for line in f]
        print(f"Loaded {len(existing_candidates)} existing candidates")
    except FileNotFoundError:
        print(f"No existing candidates file found, creating new one")
    
    # Combine existing and new candidates
    all_candidates_combined = existing_candidates + all_candidates
    
    # Save all candidates
    with open(output_file, 'w', encoding='utf-8') as f:
        for candidate in all_candidates_combined:
            f.write(json.dumps(candidate, ensure_ascii=False) + '\n')
    
    print(f"\nSaved {len(all_candidates_combined)} total candidates to {output_file}")
    print(f"   - Existing: {len(existing_candidates)}")
    print(f"   - New: {len(all_candidates)}")
    
    # Generate statistics
    stats = generator._generate_candidate_stats(all_candidates)
    generator._print_candidate_stats(stats)
    
    print(f"\nStep 2 completed successfully!")
    print(f"Output file: {output_file}")
    print(f"Ready for Step 3: Fine-tuning Ranking LLM")
    
    return all_candidates

if __name__ == "__main__":
    process_in_batches() 