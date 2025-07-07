import json
import requests
import time
import random
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
import re
import sqlite3
import pandas as pd
from collections import Counter

class CandidateAnswerGenerator:
    """Step 2: Generate candidate answers using FinMORAL framework's three specialized modules"""
    
    def __init__(self):
        # API Keys - Replace with your own keys
        self.groq_api_key = "YOUR_GROQ_API_KEY_HERE"  
        self.together_api_key = "YOUR_TOGETHER_API_KEY_HERE"  
        self.google_api_key = "YOUR_GOOGLE_API_KEY_HERE"  
        
        # Use real LLMs for generation
        self.use_mock = False
        
        # API rate limiting settings
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        
        # CoT self-consistency settings
        self.cot_samples = 5  # k = 5 generations for majority voting
    
    def generate_sql_answer(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """SQL Module: Generate SQL queries and execute them"""
        try:
            # Create SQL prompt using table schema and question
            schema = example.get('S', {})
            question = example.get('q', '')
            table_text = example.get('T', '')
            
            sql_prompt = f"""Given the following table schema and question, generate a SQL query to find the answer.

Table Schema:
{json.dumps(schema, indent=2)}

Table Data:
{table_text}

Question: {question}

Generate a SQL query to answer this question:"""
            
            # Generate SQL query using LLM
            sql_query = self._generate_sql_query(sql_prompt)
            
            # Execute SQL query
            result = self._execute_sql_query(sql_query, table_text)
            
            return {
                "type": "SQL",
                "text": str(result),
                "reasoning": f"SQL Query: {sql_query}",
                "confidence": 0.85,
                "model": "sql-module"
            }
            
        except Exception as e:
            return {
                "type": "SQL",
                "text": f"Error: {str(e)}",
                "reasoning": f"SQL generation failed: {str(e)}",
                "confidence": 0.1,
                "model": "sql-module"
            }
    
    def generate_numsolver_answer(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """NumSolver: Parse arithmetic expression trees using symbolic reasoning"""
        try:
            question = example.get('q', '')
            numbers = example.get('N', [])
            table_text = example.get('T', '')
            
            # Create arithmetic expression prompt
            numsolver_prompt = f"""Given the question and available numbers, create an arithmetic expression to solve the problem.

Question: {question}
Available numbers: {numbers}
Table context: {table_text[:200]}...

Create a mathematical expression using the available numbers to answer the question:"""
            
            # Generate arithmetic expression
            expression = self._generate_arithmetic_expression(numsolver_prompt)
            
            # Evaluate the expression
            result = self._evaluate_expression(expression, numbers)
            
            return {
                "type": "NumSolver",
                "text": str(result),
                "reasoning": f"Expression: {expression}",
                "confidence": 0.9,
                "model": "numsolver"
            }
            
        except Exception as e:
            return {
                "type": "NumSolver",
                "text": f"Error: {str(e)}",
                "reasoning": f"NumSolver failed: {str(e)}",
                "confidence": 0.1,
                "model": "numsolver"
            }
    
    def generate_cot_answer(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """CoT Reasoning: Step-by-step reasoning with self-consistency voting"""
        try:
            question = example.get('q', '')
            table_text = example.get('T', '')
            passage = example.get('P', '')
            numbers = example.get('N', [])
            
            # Create CoT prompt
            cot_prompt = f"""Answer the following question step by step using the provided table and context.

Table:
{table_text}

Context: {passage}

Question: {question}

Available numbers: {numbers}

Please provide a step-by-step reasoning to answer this question:"""
            
            # Generate multiple CoT samples for self-consistency
            cot_samples = []
            for i in range(self.cot_samples):
                sample = self._generate_cot_sample(cot_prompt)
                cot_samples.append(sample)
            
            # Apply majority voting
            final_answer = self._majority_vote_cot(cot_samples)
            
            return {
                "type": "CoT",
                "text": final_answer,
                "reasoning": f"CoT with {self.cot_samples} samples, majority voting applied",
                "confidence": 0.8,
                "model": "cot-reasoning"
            }
            
        except Exception as e:
            return {
                "type": "CoT",
                "text": f"Error: {str(e)}",
                "reasoning": f"CoT generation failed: {str(e)}",
                "confidence": 0.1,
                "model": "cot-reasoning"
            }
    
    def _generate_sql_query(self, prompt: str) -> str:
        """Generate SQL query using LLM"""
        if self.use_mock:
            return "SELECT COUNT(*) FROM table WHERE condition = 'value'"
        
        headers = {"Authorization": f"Bearer {self.together_api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "deepseek-ai/DeepSeek-V3",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content'].strip()
            else:
                return "SELECT * FROM table LIMIT 1"
                
        except Exception as e:
            print(f"SQL generation error: {e}")
            return "SELECT * FROM table LIMIT 1"
    
    def _execute_sql_query(self, sql_query: str, table_text: str) -> str:
        """Execute SQL query on table data"""
        try:
            # Parse table text into DataFrame
            lines = table_text.strip().split('\n')
            if len(lines) < 2:
                return "No table data"
            
            # Extract headers and data
            headers = lines[0].strip('|').split('|')
            data = []
            for line in lines[1:]:
                if line.strip():
                    row = line.strip('|').split('|')
                    data.append(row)
            
            # Create DataFrame
            if data:
                df = pd.DataFrame(data, columns=headers[:len(data[0])])
            else:
                df = pd.DataFrame(columns=headers)
            
            # Create in-memory SQLite database
            conn = sqlite3.connect(':memory:')
            df.to_sql('table', conn, index=False, if_exists='replace')
            
            # Execute query
            result = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            # Return first value if single result
            if len(result) == 1 and len(result.columns) == 1:
                return str(result.iloc[0, 0])
            else:
                return str(result.to_dict('records'))
                
        except Exception as e:
            return f"SQL execution error: {str(e)}"
    
    def _generate_arithmetic_expression(self, prompt: str) -> str:
        """Generate arithmetic expression using LLM"""
        if self.use_mock:
            return "100 + 50"
        
        headers = {"Authorization": f"Bearer {self.together_api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "deepseek-ai/DeepSeek-V3",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content'].strip()
            else:
                return "0"
                
        except Exception as e:
            print(f"Arithmetic generation error: {e}")
            return "0"
    
    def _evaluate_expression(self, expression: str, numbers: List[str]) -> str:
        """Evaluate arithmetic expression safely"""
        try:
            # Clean expression and replace with actual numbers
            clean_expr = re.sub(r'[^0-9+\-*/().]', '', expression)
            
            # Basic safety check
            if any(op in clean_expr for op in ['import', 'eval', 'exec']):
                return "0"
            
            # Evaluate expression
            result = eval(clean_expr)
            return str(result)
            
        except Exception as e:
            return f"Evaluation error: {str(e)}"
    
    def _generate_cot_sample(self, prompt: str) -> str:
        """Generate a single CoT sample"""
        if self.use_mock:
            return "Step 1: Analyze the table\nStep 2: Find relevant data\nStep 3: Calculate answer\nAnswer: 42"
        
        headers = {"Authorization": f"Bearer {self.together_api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "deepseek-ai/DeepSeek-V3",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.3
        }
        
        try:
            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content'].strip()
            else:
                return "No reasoning generated"
                
        except Exception as e:
            print(f"CoT generation error: {e}")
            return "Error in reasoning"
    
    def _majority_vote_cot(self, samples: List[str]) -> str:
        """Apply majority voting to CoT samples"""
        try:
            # Extract final answers from samples
            answers = []
            for sample in samples:
                # Look for patterns like "Answer: X" or "Final answer: X"
                answer_match = re.search(r'(?:Answer|Final answer|Result):\s*([^\n]+)', sample, re.IGNORECASE)
                if answer_match:
                    answers.append(answer_match.group(1).strip())
                else:
                    # Take the last line as answer
                    lines = sample.strip().split('\n')
                    if lines:
                        answers.append(lines[-1].strip())
            
            if not answers:
                return "No valid answers found"
            
            # Count occurrences
            answer_counts = Counter(answers)
            
            # Return most common answer
            return answer_counts.most_common(1)[0][0]
            
        except Exception as e:
            return f"Majority voting error: {str(e)}"
    
    def generate_candidates_for_example(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate candidates using all three FinMORAL modules"""
        candidates = []
        
        # Generate SQL answer
        sql_candidate = self.generate_sql_answer(example)
        candidates.append(sql_candidate)
        
        # Generate NumSolver answer
        numsolver_candidate = self.generate_numsolver_answer(example)
        candidates.append(numsolver_candidate)
        
        # Generate CoT answer
        cot_candidate = self.generate_cot_answer(example)
        candidates.append(cot_candidate)
        
        return candidates
    
    def process_dataset(self, input_file: str, output_file: str, max_examples: int = 100):
        """Process entire dataset and generate candidates"""
        print(f"Processing dataset: {input_file}")
        print(f"Max examples: {max_examples}")
        
        # Load processed data
        examples = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_examples:
                    break
                examples.append(json.loads(line))
        
        print(f"Loaded {len(examples)} examples")
        
        # Generate candidates for each example
        all_candidates = []
        
        for i, example in enumerate(tqdm(examples, desc="Generating candidates")):
            example_id = example.get('id', f'example_{i}')
            
            # Generate candidates
            candidates = self.generate_candidates_for_example(example)
            
            # Add metadata to each candidate
            for candidate in candidates:
                candidate['example_id'] = example_id
                candidate['dataset'] = example.get('dataset', 'unknown')
                candidate['question'] = example.get('q', '')
                candidate['gold_answer'] = example.get('answer', '')
            
            all_candidates.extend(candidates)
            
            # Rate limiting
            if i < len(examples) - 1:
                time.sleep(0.5)
        
        # Save candidates
        with open(output_file, 'w', encoding='utf-8') as f:
            for candidate in all_candidates:
                f.write(json.dumps(candidate, ensure_ascii=False) + '\n')
        
        print(f"Generated {len(all_candidates)} candidates")
        print(f"Saved to: {output_file}")
        
        # Print statistics
        self._print_candidate_stats(all_candidates)
        
        return all_candidates
    
    def _print_candidate_stats(self, candidates: List[Dict[str, Any]]):
        """Print statistics about generated candidates"""
        print("\nCandidate Generation Statistics:")
        print("=" * 40)
        
        # Count by type
        type_counts = {}
        for candidate in candidates:
            candidate_type = candidate['type']
            type_counts[candidate_type] = type_counts.get(candidate_type, 0) + 1
        
        for candidate_type, count in type_counts.items():
            print(f"{candidate_type}: {count} candidates")
        
        # Average confidence by type
        confidence_by_type = {}
        for candidate in candidates:
            candidate_type = candidate['type']
            if candidate_type not in confidence_by_type:
                confidence_by_type[candidate_type] = []
            confidence_by_type[candidate_type].append(candidate.get('confidence', 0))
        
        print("\nAverage confidence by type:")
        for candidate_type, confidences in confidence_by_type.items():
            avg_confidence = sum(confidences) / len(confidences)
            print(f"{candidate_type}: {avg_confidence:.3f}")

def main():
    """Main function for Step 2: Candidate generation following FinMORAL framework"""
    print("Step 2: Candidate Answer Generation for FinMORAL Framework")
    print("=" * 60)
    
    # Initialize generator
    generator = CandidateAnswerGenerator()
    
    # Process dataset
    input_file = "step1_processed_data.jsonl"
    output_file = "step2_candidates.jsonl"
    
    # Generate candidates
    candidates = generator.process_dataset(input_file, output_file, max_examples=50)
    
    print(f"\nStep 2 completed successfully!")
    print(f"Output file: {output_file}")
    print(f"Ready for Step 3: Final Answer Selection (Reranking)")

if __name__ == "__main__":
    main() 