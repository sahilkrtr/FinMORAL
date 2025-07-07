import json
import pandas as pd
import os
from typing import Dict, List, Any, Tuple
import re
from pathlib import Path
from tqdm import tqdm

class DataPreparationStep1:
    """Step 1: Data preparation and prompt formatting for FinMORAL framework"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.wtq_path = self.base_path / "WTQdata"
        self.fetaqa_path = self.base_path / "fetaQAdata"
        
    def flatten_table_to_text(self, table: List[List[str]]) -> str:
        """Convert table to flattened text format with | separators for columns, \n for rows"""
        if not table or len(table) == 0:
            return ""
        
        # Get headers (first row)
        headers = table[0]
        header_row = "|" + "|".join(str(h) for h in headers) + "|"
        
        # Get data rows
        data_rows = []
        for row in table[1:]:
            if len(row) == len(headers):  # Ensure row matches header length
                data_row = "|" + "|".join(str(cell) for cell in row) + "|"
                data_rows.append(data_row)
        
        # Combine header and data
        table_text = header_row + "\n" + "\n".join(data_rows)
        return table_text
    
    def extract_numbers_from_text(self, text: str) -> List[str]:
        """Extract numbers from text for number list (N)"""
        numbers = re.findall(r'\b\d+\.?\d*\b', text)
        return numbers
    
    def create_schema_metadata(self, table: List[List[str]]) -> Dict[str, Any]:
        """Create schema metadata (S) from table structure"""
        if not table or len(table) == 0:
            return {}
        
        headers = table[0]
        schema = {
            "num_columns": len(headers),
            "num_rows": len(table) - 1,
            "column_names": headers,
            "column_types": self._infer_column_types(table)
        }
        return schema
    
    def _infer_column_types(self, table: List[List[str]]) -> List[str]:
        """Infer column types based on content"""
        if len(table) < 2:
            return ["text"] * len(table[0]) if table else []
        
        headers = table[0]
        types = []
        
        for col_idx in range(len(headers)):
            col_values = [row[col_idx] for row in table[1:] if col_idx < len(row)]
            
            # Check if column contains mostly numbers
            numeric_count = 0
            for val in col_values:
                try:
                    float(re.sub(r'[$,%,\s]', '', str(val)))
                    numeric_count += 1
                except:
                    pass
            
            if numeric_count > len(col_values) * 0.7:  # 70% threshold
                types.append("numeric")
            else:
                types.append("text")
        
        return types
    
    def load_wtq_data(self, split: str = "training", max_examples: int = 1000) -> List[Dict[str, Any]]:
        """Load and preprocess WTQ data according to FinMORAL framework"""
        file_path = self.wtq_path / f"{split}.tsv"
        
        # Read TSV file
        df = pd.read_csv(file_path, sep='\t')
        
        processed_examples = []
        
        for _, row in tqdm(df.head(max_examples).iterrows(), desc="Processing WTQ"):
            question_id = row['id']
            question = row['utterance']
            context = row['context']
            answer = row['targetValue']
            # Ensure context is a string (fix linter error)
            if not isinstance(context, str):
                context = str(context)
            # Load the actual table file
            table_text = self._load_wtq_table(context)
            
            # Extract table structure for schema
            table_array = self._parse_table_text(table_text)
            schema = self.create_schema_metadata(table_array)
            
            # Extract numbers from question and answer
            numbers = self.extract_numbers_from_text(str(question) + " " + str(answer))
            
            # Create structured example following FinMORAL format
            processed_example = {
                'id': question_id,
                'dataset': 'wtq',
                'split': split,
                'q': question,  # Question
                'T': table_text,  # Table
                'P': '',  # Passage (empty for WTQ)
                'N': numbers,  # Number list
                'S': schema,  # Schema metadata
                'answer': answer,
                'answer_type': 'numeric' if self._is_numeric(str(answer)) else 'text'
            }
            
            processed_examples.append(processed_example)
        
        return processed_examples
    
    def clean_ftq_answer_critical(self, data: dict) -> dict:
        """Placeholder: Clean FTQ entry to retain only answer-critical content (FinMORAL step). In practice, use GPT-4 + human check."""
        # TODO: Implement answer-critical content filtering using LLM + human if available
        # For now, return as-is
        return data

    def load_ftq_data(self, split: str = "dev", max_examples: int = 1000) -> List[Dict[str, Any]]:
        """Load and preprocess FTQ (Filtered FeTaQA) data according to FinMORAL framework"""
        file_path = self.fetaqa_path / f"fetaQA-v1_{split}.jsonl"
        processed_examples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Processing FTQ")):
                if i >= max_examples:
                    break
                data = json.loads(line)
                # --- FinMORAL: Clean FTQ for answer-critical content ---
                data = self.clean_ftq_answer_critical(data)
                # Extract components
                question = data.get('question', '')
                answer = data.get('answer', '')
                table_array = data.get('table_array', [])
                # Convert table to text format
                table_text = self.flatten_table_to_text(table_array)
                # Create schema metadata
                schema = self.create_schema_metadata(table_array)
                # Extract numbers from question and answer
                numbers = self.extract_numbers_from_text(question + " " + str(answer))
                # Create passage from context (if available)
                passage = ""
                if 'page_wikipedia_url' in data:
                    passage = f"Context from: {data['page_wikipedia_url']}"
                # Create structured example following FinMORAL format
                processed_example = {
                    'id': data.get('feta_id', f'ftq_{i}'),
                    'dataset': 'ftq',
                    'split': split,
                    'q': question,  # Question
                    'T': table_text,  # Table
                    'P': passage,  # Passage
                    'N': numbers,  # Number list
                    'S': schema,  # Schema metadata
                    'answer': answer,
                    'answer_type': 'numeric' if self._is_numeric(answer) else 'text'
                }
                processed_examples.append(processed_example)
        return processed_examples
    
    def _load_wtq_table(self, context: str) -> str:
        """Load actual table file for WTQ"""
        try:
            table_path = self.wtq_path / context
            if table_path.exists():
                # Read CSV file
                df = pd.read_csv(table_path)
                # Convert to list of lists
                table = [df.columns.tolist()] + df.values.tolist()
                return self.flatten_table_to_text(table)
            else:
                return f"Table file not found: {context}"
        except Exception as e:
            return f"Error loading table {context}: {str(e)}"
    
    def _parse_table_text(self, table_text: str) -> List[List[str]]:
        """Parse table text back to array format"""
        lines = table_text.strip().split('\n')
        table = []
        for line in lines:
            if line.strip():
                # Remove leading/trailing | and split by |
                cells = line.strip('|').split('|')
                table.append(cells)
        return table
    
    def _is_numeric(self, value: str) -> bool:
        """Check if a value is numeric"""
        try:
            # Remove common non-numeric characters
            cleaned = re.sub(r'[$,%,\s]', '', str(value))
            float(cleaned)
            return True
        except:
            return False
    
    def save_processed_data(self, data: List[Dict[str, Any]], output_path: str):
        """Save processed data to JSONL format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    def get_dataset_stats(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get comprehensive statistics about the processed dataset"""
        stats = {
            'total_examples': len(data),
            'datasets': {},
            'answer_types': {},
            'avg_question_length': 0,
            'avg_table_rows': 0,
            'avg_table_cols': 0,
            'avg_numbers_per_example': 0,
            'schema_types': {}
        }
        
        question_lengths = []
        table_rows = []
        table_cols = []
        numbers_per_example = []
        
        for example in data:
            # Dataset distribution
            dataset = example['dataset']
            stats['datasets'][dataset] = stats['datasets'].get(dataset, 0) + 1
            
            # Answer type distribution
            answer_type = example['answer_type']
            stats['answer_types'][answer_type] = stats['answer_types'].get(answer_type, 0) + 1
            
            # Question length
            question_lengths.append(len(example['q']))
            
            # Table dimensions
            table_lines = example['T'].count('\n') + 1
            if table_lines > 0:
                first_line = example['T'].split('\n')[0]
                cols = first_line.count('|') - 1
                table_rows.append(table_lines)
                table_cols.append(cols)
            
            # Numbers per example
            numbers_per_example.append(len(example['N']))
            
            # Schema types
            schema = example.get('S', {})
            for col_type in schema.get('column_types', []):
                stats['schema_types'][col_type] = stats['schema_types'].get(col_type, 0) + 1
        
        if question_lengths:
            stats['avg_question_length'] = sum(question_lengths) / len(question_lengths)
        if table_rows:
            stats['avg_table_rows'] = sum(table_rows) / len(table_rows)
        if table_cols:
            stats['avg_table_cols'] = sum(table_cols) / len(table_cols)
        if numbers_per_example:
            stats['avg_numbers_per_example'] = sum(numbers_per_example) / len(numbers_per_example)
        
        return stats

def main():
    """Main function for Step 1: Data preparation following FinMORAL framework"""
    print("Step 1: Data Preparation for FinMORAL Framework")
    print("=" * 60)
    
    # Initialize processor
    processor = DataPreparationStep1()
    
    # Process each dataset
    all_data = []
    
    # WTQ Dataset
    print("\nProcessing WTQ dataset...")
    try:
        wtq_data = processor.load_wtq_data("training", max_examples=1000)
        all_data.extend(wtq_data)
        print(f"Loaded {len(wtq_data)} WTQ examples")
    except Exception as e:
        print(f"Error loading WTQ: {e}")
    
    # FTQ Dataset (Filtered FeTaQA)
    print("\nProcessing FTQ dataset...")
    try:
        ftq_data = processor.load_ftq_data("dev", max_examples=1000)
        all_data.extend(ftq_data)
        print(f"Loaded {len(ftq_data)} FTQ examples")
    except Exception as e:
        print(f"Error loading FTQ: {e}")
    
    # Save processed data
    output_path = "step1_processed_data.jsonl"
    processor.save_processed_data(all_data, output_path)
    print(f"\nSaved {len(all_data)} processed examples to {output_path}")
    
    # Get and display statistics
    stats = processor.get_dataset_stats(all_data)
    print(f"\nDataset Statistics:")
    print(f"Total examples: {stats['total_examples']}")
    print(f"Dataset distribution: {stats['datasets']}")
    print(f"Answer types: {stats['answer_types']}")
    print(f"Average question length: {stats['avg_question_length']:.1f} characters")
    print(f"Average table size: {stats['avg_table_rows']:.1f} rows × {stats['avg_table_cols']:.1f} columns")
    print(f"Average numbers per example: {stats['avg_numbers_per_example']:.1f}")
    print(f"Schema types: {stats['schema_types']}")
    
    # Show sample examples
    print(f"\nSample processed examples:")
    for i, example in enumerate(all_data[:3]):
        print(f"\n--- Example {i+1} ({example['dataset']}) ---")
        print(f"Question (q): {example['q']}")
        print(f"Answer: {example['answer']}")
        print(f"Answer Type: {example['answer_type']}")
        print(f"Numbers (N): {example['N']}")
        print(f"Schema (S): {example['S']}")
        print(f"Table preview: {example['T'][:100]}...")
        if example['P']:
            print(f"Passage (P): {example['P']}")
    
    print(f"\nStep 1 completed successfully!")
    print(f"Output file: {output_path}")
    print(f"Ready for Step 2: Candidate Answer Generation")

if __name__ == "__main__":
    main() 