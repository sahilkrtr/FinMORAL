import json
import time
import random
from typing import List, Dict, Any
from tqdm import tqdm
import os
from step2_candidate_generation import CandidateAnswerGenerator

def load_processed_data(file_path: str) -> List[Dict[str, Any]]:
    """Load processed data from Step 1"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_candidates_batch(candidates: List[Dict], output_file: str, mode: str = 'a'):
    """Save candidates in batches to avoid memory issues"""
    with open(output_file, mode, encoding='utf-8') as f:
        for candidate in candidates:
            f.write(json.dumps(candidate, ensure_ascii=False) + '\n')

def process_full_dataset():
    """Process the full dataset with optimized batching"""
    print("Step 2: Full Dataset Candidate Generation")
    print("=" * 60)
    
    # Initialize the generator
    generator = CandidateAnswerGenerator()
    
    # Load full dataset
    data_file = "step1_processed_data.jsonl"
    all_examples = load_processed_data(data_file)
    
    print(f"Total examples available: {len(all_examples)}")
    
    # Group by dataset
    datasets = {}
    for example in all_examples:
        dataset = example.get('dataset', 'unknown')
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(example)
    
    print(f"Datasets found: {list(datasets.keys())}")
    for dataset, examples in datasets.items():
        print(f"   {dataset}: {len(examples)} examples")
    
    # Configuration for full dataset
    BATCH_SIZE = 5  # Smaller batches for stability
    DELAY_BETWEEN_BATCHES = 15  # Longer delays to avoid rate limits
    SAVE_INTERVAL = 50  # Save every 50 candidates
    
    output_file = "step2_candidates_full.jsonl"
    all_candidates = []
    total_processed = 0
    
    # Process each dataset
    for dataset_name, examples in datasets.items():
        print(f"\nPROCESSING DATASET: {dataset_name.upper()}")
        print("=" * 60)
        
        # Process in batches
        num_batches = (len(examples) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for batch_idx in tqdm(range(num_batches), desc=f"Processing {dataset_name}"):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(examples))
            batch_examples = examples[start_idx:end_idx]
            
            batch_candidates = []
            
            for example in batch_examples:
                question_id = example['id']
                question = example['question']
                gold_answer = example['answer']
                dataset = example['dataset']
                
                # Create prompt for the question
                prompt = f"Question: {question}\nAnswer: {gold_answer}"
                
                # Generate candidates for each strategy
                strategies = [
                    ('SQL', generator.generate_sql_answer),
                    ('CoT', generator.generate_cot_answer), 
                    ('Mix-SC', generator.generate_mix_sc_answer),
                    ('NumSolver', generator.generate_numsolver_answer),
                    ('Random', lambda p: {
                        'type': 'Random',
                        'text': generator.generate_random_distractor(gold_answer, 'number'),
                        'reasoning': 'Randomly generated distractor',
                        'confidence': random.uniform(0.1, 0.4),
                        'model': 'N/A'
                    })
                ]
                
                for strategy_name, strategy_func in strategies:
                    try:
                        # Add retry logic for API calls
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                candidate = strategy_func(prompt)
                                candidate['question_id'] = question_id
                                candidate['dataset'] = dataset
                                candidate['gold_answer'] = gold_answer
                                batch_candidates.append(candidate)
                                break
                            except Exception as e:
                                if attempt < max_retries - 1:
                                    wait_time = (attempt + 1) * 5
                                    print(f"{strategy_name} failed, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                                    time.sleep(wait_time)
                                else:
                                    print(f"{strategy_name} failed after {max_retries} attempts: {e}")
                                    # Add fallback candidate
                                    fallback = {
                                        'question_id': question_id,
                                        'dataset': dataset,
                                        'gold_answer': gold_answer,
                                        'text': f"Error in {strategy_name}",
                                        'confidence': 0.1,
                                        'type': strategy_name,
                                        'reasoning': f"Failed to generate: {str(e)}"
                                    }
                                    batch_candidates.append(fallback)
                    except Exception as e:
                        print(f"Error with {strategy_name}: {e}")
            
            # Save batch candidates
            save_candidates_batch(batch_candidates, output_file, 'a')
            all_candidates.extend(batch_candidates)
            total_processed += len(batch_examples)
            
            # Progress update
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {total_processed}/{len(all_examples)} examples ({total_processed/len(all_examples)*100:.1f}%)")
                print(f"Generated {len(all_candidates)} candidates so far")
            
            # Delay between batches
            if batch_idx < num_batches - 1:
                print(f"Waiting {DELAY_BETWEEN_BATCHES} seconds before next batch...")
                time.sleep(DELAY_BETWEEN_BATCHES)
    
    print(f"\nFULL DATASET PROCESSING COMPLETED!")
    print(f"Total examples processed: {len(all_examples)}")
    print(f"Total candidates generated: {len(all_candidates)}")
    print(f"Results saved to: {output_file}")
    
    # Final statistics
    strategy_counts = {}
    for candidate in all_candidates:
        strategy = candidate['type']
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    print(f"\nSTRATEGY DISTRIBUTION:")
    for strategy, count in strategy_counts.items():
        percentage = (count / len(all_candidates)) * 100
        print(f"   {strategy}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    process_full_dataset() 