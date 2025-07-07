import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
import random
from collections import Counter

class MixSCVoting:
    """Mix-SC Self-Consistency Voting for FinMORAL framework"""
    
    def __init__(self, lambda_weight: float = 0.5):
        self.lambda_weight = lambda_weight
    
    def calculate_consistency_score(self, candidates: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate consistency score based on majority agreement"""
        consistency_scores = {}
        
        # Extract answers from candidates
        answers = [c['text'] for c in candidates]
        
        # Count occurrences of each answer
        answer_counts = Counter(answers)
        total_candidates = len(candidates)
        
        for i, candidate in enumerate(candidates):
            answer = candidate['text']
            count = answer_counts[answer]
            # Consistency score is the proportion of candidates with the same answer
            consistency_scores[i] = count / total_candidates
        
        return consistency_scores
    
    def calculate_heuristic_score(self, candidates: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate heuristic trust score H(ai)"""
        heuristic_scores = {}
        
        for i, candidate in enumerate(candidates):
            score = 0.0
            
            # SQL execution success
            if candidate['type'] == 'SQL':
                if 'Error' not in candidate['text']:
                    score += 0.3
                if 'SELECT' in candidate.get('reasoning', ''):
                    score += 0.2
            
            # CoT alignment (check if reasoning is coherent)
            elif candidate['type'] == 'CoT':
                reasoning = candidate.get('reasoning', '')
                if len(reasoning) > 50:  # Reasonable length
                    score += 0.3
                if 'step' in reasoning.lower():
                    score += 0.2
            
            # NumSolver confidence
            elif candidate['type'] == 'NumSolver':
                if candidate.get('confidence', 0) > 0.8:
                    score += 0.3
                if 'Expression' in candidate.get('reasoning', ''):
                    score += 0.2
            
            # Base confidence from model
            score += candidate.get('confidence', 0.5) * 0.3
            
            heuristic_scores[i] = min(score, 1.0)  # Cap at 1.0
        
        return heuristic_scores
    
    def combine_scores(self, consistency_scores: Dict[str, float], 
                      heuristic_scores: Dict[str, float]) -> Dict[str, float]:
        """Combine consistency and heuristic scores: C(ai) = consistency_score + λ * heuristic_score"""
        combined_scores = {}
        
        for i in consistency_scores:
            consistency = consistency_scores[i]
            heuristic = heuristic_scores.get(i, 0.0)
            combined = consistency + self.lambda_weight * heuristic
            combined_scores[i] = combined
        
        return combined_scores
    
    def select_best_candidate(self, candidates: List[Dict[str, Any]]) -> int:
        """Select the best candidate using Mix-SC voting"""
        if not candidates:
            return -1
        
        # Calculate scores
        consistency_scores = self.calculate_consistency_score(candidates)
        heuristic_scores = self.calculate_heuristic_score(candidates)
        combined_scores = self.combine_scores(consistency_scores, heuristic_scores)
        
        # Find candidate with highest combined score
        best_idx = max(combined_scores.keys(), key=lambda k: combined_scores[k])
        
        return best_idx

class PairwiseRerankerDataset(Dataset):
    """Dataset for pairwise reranking following FinMORAL framework"""
    
    def __init__(self, examples: List[Dict[str, Any]], tokenizer, max_length: int = 256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Create input text for pairwise ranking: [question; context; answer_i; answer_j]
        input_text = self._create_pairwise_input(example)
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_attention_mask=True
        )
        
        # Get label (1 if answer_i is better, 0 if answer_j is better)
        label = 1 if example['answer_i_better'] else 0
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    def _create_pairwise_input(self, example: Dict[str, Any]) -> str:
        """Create input text for pairwise ranking: [question; context; answer_i; answer_j]"""
        question = example['question']
        context = example.get('context', '')
        answer_i = example['answer_i']
        answer_j = example['answer_j']
        
        # Format: [question; context; answer_i; answer_j]
        input_text = f"Question: {question}\nContext: {context}\nAnswer A: {answer_i}\nAnswer B: {answer_j}"
        
        return input_text

class DistilBERTReranker(nn.Module):
    """DistilBERT-based pairwise reranker for FinMORAL framework"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2  # Binary classification: answer_i better or answer_j better
        )
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

class FinMORALReranker:
    """FinMORAL Reranker combining Mix-SC voting and DistilBERT pairwise ranking"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = DistilBERTReranker(model_name).to(self.device)
        self.mix_sc = MixSCVoting()
        
        # Fix padding token issues
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Ensure padding is enabled
        self.tokenizer.padding_side = 'right'
        
        print(f"Using device: {self.device}")
        print(f"Model: {model_name}")
    
    def prepare_pairwise_training_data(self, candidates_file: str) -> List[Dict[str, Any]]:
        """Prepare pairwise training data from candidates"""
        print("Preparing pairwise training data...")
        
        # Load candidates
        candidates = []
        with open(candidates_file, 'r', encoding='utf-8') as f:
            for line in f:
                candidates.append(json.loads(line))
        
        # Group candidates by example
        example_groups = {}
        for candidate in candidates:
            example_id = candidate['example_id']
            if example_id not in example_groups:
                example_groups[example_id] = []
            example_groups[example_id].append(candidate)
        
        # Create pairwise examples
        pairwise_examples = []
        
        for example_id, example_candidates in example_groups.items():
            if len(example_candidates) < 2:
                continue
            
            # Get question and context from first candidate
            first_candidate = example_candidates[0]
            question = first_candidate.get('question', '')
            gold_answer = first_candidate.get('gold_answer', '')
            
            # Create all pairwise combinations
            for i in range(len(example_candidates)):
                for j in range(i + 1, len(example_candidates)):
                    candidate_i = example_candidates[i]
                    candidate_j = example_candidates[j]
                    
                    # Determine which answer is better (closer to gold)
                    score_i = self._calculate_answer_similarity(candidate_i['text'], gold_answer)
                    score_j = self._calculate_answer_similarity(candidate_j['text'], gold_answer)
                    
                    answer_i_better = score_i > score_j
                    
                    # Create pairwise example
                    pairwise_example = {
                        'question': question,
                        'context': '',  # Could be enhanced with table context
                        'answer_i': candidate_i['text'],
                        'answer_j': candidate_j['text'],
                        'answer_i_better': answer_i_better,
                        'candidate_i_type': candidate_i['type'],
                        'candidate_j_type': candidate_j['type']
                    }
                    
                    pairwise_examples.append(pairwise_example)
        
        print(f"Created {len(pairwise_examples)} pairwise examples")
        
        return pairwise_examples
    
    def _calculate_answer_similarity(self, candidate_answer: str, gold_answer: str) -> float:
        """Calculate similarity between candidate and gold answer"""
        # Simple exact match for now
        # Could be enhanced with semantic similarity
        candidate_clean = candidate_answer.strip().lower()
        gold_clean = gold_answer.strip().lower()
        
        if candidate_clean == gold_clean:
            return 1.0
        
        # Check if gold answer is contained in candidate
        if gold_clean in candidate_clean:
            return 0.8
        
        # Check if candidate is contained in gold
        if candidate_clean in gold_clean:
            return 0.6
        
        return 0.0
    
    def train(self, training_examples: List[Dict[str, Any]], 
              batch_size: int = 8, epochs: int = 3, learning_rate: float = 2e-5):
        """Train the DistilBERT pairwise reranker"""
        print(f"Training DistilBERT pairwise reranker...")
        print(f"Training examples: {len(training_examples)}")
        print(f"Batch size: {batch_size}, Epochs: {epochs}")
        
        # Split data
        train_examples, val_examples = train_test_split(
            training_examples, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = PairwiseRerankerDataset(train_examples, self.tokenizer)
        val_dataset = PairwiseRerankerDataset(val_examples, self.tokenizer)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup training
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.logits, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            train_acc = train_correct / train_total if train_total > 0 else 0
            val_acc = val_correct / val_total if val_total > 0 else 0
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model("best_reranker_model.pth")
                print(f"  New best model saved! (Val Acc: {val_acc:.4f})")
    
    def predict_pairwise_preference(self, question: str, answer_i: str, answer_j: str, 
                                  context: str = "") -> float:
        """Predict pairwise preference score σij = frank([q; C; ai; aj])"""
        # Create input
        input_text = f"Question: {question}\nContext: {context}\nAnswer A: {answer_i}\nAnswer B: {answer_j}"
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt',
            return_attention_mask=True
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            # Return probability that answer_i is better
            preference_score = probs[0, 1].item()
        
        return preference_score
    
    def select_final_answer(self, question: str, candidates: List[Dict[str, Any]], 
                          context: str = "") -> Dict[str, Any]:
        """Select final answer using FinMORAL approach: Mix-SC + DistilBERT reranking"""
        
        # Step 1: Apply Mix-SC voting
        mix_sc_best_idx = self.mix_sc.select_best_candidate(candidates)
        
        # Step 2: Apply DistilBERT pairwise reranking
        if len(candidates) > 1:
            # Calculate total preference score for each candidate
            preference_scores = []
            for i, candidate_i in enumerate(candidates):
                total_score = 0.0
                for j, candidate_j in enumerate(candidates):
                    if i != j:
                        # Get pairwise preference score
                        pref_score = self.predict_pairwise_preference(
                            question, candidate_i['text'], candidate_j['text'], context
                        )
                        total_score += pref_score
                
                preference_scores.append(total_score)
            
            # Select candidate with highest total preference score
            reranker_best_idx = np.argmax(preference_scores)
        else:
            reranker_best_idx = 0
        
        # Combine both approaches (could be weighted)
        final_idx = reranker_best_idx  # Prefer reranker for now
        
        selected_candidate = candidates[final_idx]
        selected_candidate['mix_sc_idx'] = mix_sc_best_idx
        selected_candidate['reranker_idx'] = reranker_best_idx
        selected_candidate['final_idx'] = final_idx
        
        return selected_candidate
    
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")

def main():
    """Main function for Step 3: Final Answer Selection following FinMORAL framework"""
    print("Step 3: Final Answer Selection (Reranking) for FinMORAL Framework")
    print("=" * 60)
    
    # Initialize reranker
    reranker = FinMORALReranker()
    
    # Prepare training data
    candidates_file = "step2_candidates.jsonl"
    training_examples = reranker.prepare_pairwise_training_data(candidates_file)
    
    if len(training_examples) > 0:
        # Train the reranker
        reranker.train(training_examples, batch_size=8, epochs=3)
        
        # Test on a few examples
        print("\nTesting reranker on sample candidates...")
        
        # Load some candidates for testing
        with open(candidates_file, 'r', encoding='utf-8') as f:
            candidates = [json.loads(line) for line in f]
        
        # Group by example
        example_groups = {}
        for candidate in candidates:
            example_id = candidate['example_id']
            if example_id not in example_groups:
                example_groups[example_id] = []
            example_groups[example_id].append(candidate)
        
        # Test on first few examples
        for i, (example_id, example_candidates) in enumerate(list(example_groups.items())[:3]):
            if len(example_candidates) < 2:
                continue
            
            question = example_candidates[0].get('question', '')
            gold_answer = example_candidates[0].get('gold_answer', '')
            
            print(f"\nExample {i+1}: {example_id}")
            print(f"Question: {question}")
            print(f"Gold Answer: {gold_answer}")
            
            # Select final answer
            final_answer = reranker.select_final_answer(question, example_candidates)
            
            print(f"Selected Answer: {final_answer['text']} (Type: {final_answer['type']})")
            print(f"Mix-SC Index: {final_answer['mix_sc_idx']}")
            print(f"Reranker Index: {final_answer['reranker_idx']}")
    
    print(f"\nStep 3 completed successfully!")
    print(f"Ready for Step 4: Evaluation")

if __name__ == "__main__":
    main() 