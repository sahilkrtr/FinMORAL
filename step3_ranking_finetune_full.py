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
import gc

class RankingDataset(Dataset):
    """Dataset for ranking candidate answers - optimized for large datasets"""
    
    def __init__(self, examples: List[Dict[str, Any]], tokenizer, max_length: int = 256):  # Reduced max_length
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Create input text for ranking
        input_text = self._create_ranking_input(example)
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_attention_mask=True
        )
        
        # Get label (1 for correct, 0 for incorrect)
        label = 1 if example['is_correct'] else 0
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    def _create_ranking_input(self, example: Dict[str, Any]) -> str:
        """Create input text for ranking model - simplified for efficiency"""
        question = example['question'][:100]  # Truncate for efficiency
        candidate_answer = example['candidate_text'][:50]  # Truncate
        gold_answer = example['gold_answer'][:50]  # Truncate
        
        # Simplified input format
        input_text = f"Q: {question} A: {candidate_answer} G: {gold_answer}"
        
        return input_text

class RankingModel(nn.Module):
    """Optimized ranking model using smaller transformer"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2  # Binary classification: correct/incorrect
        )
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

class RankingTrainer:
    """Enhanced trainer for large datasets"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = RankingModel(model_name).to(self.device)
        
        # Fix padding token issues
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Ensure padding is enabled
        self.tokenizer.padding_side = 'right'
        
        print(f"Using device: {self.device}")
        print(f"Model: {model_name}")
        print(f"Padding token: {self.tokenizer.pad_token}")
    
    def prepare_training_data(self, candidates_file: str, max_examples: int = None) -> List[Dict[str, Any]]:
        """Prepare training data from candidates - with sampling for large datasets"""
        print("Preparing training data...")
        
        # Load candidates
        candidates = []
        with open(candidates_file, 'r', encoding='utf-8') as f:
            for line in f:
                candidates.append(json.loads(line))
        
        print(f"Loaded {len(candidates)} candidates")
        
      
        # Group candidates by question
        question_groups = {}
        for candidate in candidates:
            question_id = candidate['question_id']
            if question_id not in question_groups:
                question_groups[question_id] = []
            question_groups[question_id].append(candidate)
        
        # Create training examples
        training_examples = []
        
        for question_id, question_candidates in question_groups.items():
            # Get question info from first candidate
            first_candidate = question_candidates[0]
            question = self._extract_question_from_prompt(first_candidate)
            gold_answer = first_candidate['gold_answer']
            
            # Create positive examples (correct answers)
            correct_candidates = [c for c in question_candidates if self._is_correct_answer(c['text'], gold_answer)]
            
            # Create negative examples (incorrect answers)
            incorrect_candidates = [c for c in question_candidates if not self._is_correct_answer(c['text'], gold_answer)]
            
            # Add positive examples
            for candidate in correct_candidates:
                training_examples.append({
                    'question': question,
                    'candidate_text': candidate['text'],
                    'candidate_reasoning': candidate.get('reasoning', ''),
                    'gold_answer': gold_answer,
                    'is_correct': True,
                    'confidence': candidate.get('confidence', 0.5),
                    'strategy': candidate['type']
                })
            
            # Add negative examples (limit to balance dataset)
            for candidate in incorrect_candidates[:len(correct_candidates)]:
                training_examples.append({
                    'question': question,
                    'candidate_text': candidate['text'],
                    'candidate_reasoning': candidate.get('reasoning', ''),
                    'gold_answer': gold_answer,
                    'is_correct': False,
                    'confidence': candidate.get('confidence', 0.5),
                    'strategy': candidate['type']
                })
        
        print(f"Created {len(training_examples)} training examples")
        print(f"Correct answers: {sum(1 for ex in training_examples if ex['is_correct'])}")
        print(f"Incorrect answers: {sum(1 for ex in training_examples if not ex['is_correct'])}")
        
        return training_examples
    
    def _extract_question_from_prompt(self, candidate: Dict[str, Any]) -> str:
        """Extract question from LLM prompt"""
        return f"Question from {candidate['dataset']} dataset"
    
    def _is_correct_answer(self, candidate_text: str, gold_answer: str) -> bool:
        """Check if candidate answer is correct"""
        # Simple exact match for demo
        return candidate_text.strip().lower() == gold_answer.strip().lower()
    
    def train(self, training_examples: List[Dict[str, Any]], 
              batch_size: int = 16, epochs: int = 3, learning_rate: float = 2e-5):
        """Train the ranking model with enhanced memory management"""
        print(f"Training ranking model...")
        print(f"Training examples: {len(training_examples)}")
        print(f"Batch size: {batch_size}, Epochs: {epochs}")
        
        # Split data
        train_examples, val_examples = train_test_split(
            training_examples, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = RankingDataset(train_examples, self.tokenizer)
        val_dataset = RankingDataset(val_examples, self.tokenizer)
        
        # Create dataloaders with num_workers for efficiency
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Setup training
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
        
        # Training loop
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in tqdm(train_loader, desc="Training"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs.loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(outputs.logits, dim=1)
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_predictions = []
            val_labels = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask, labels)
                    loss = outputs.loss
                    
                    val_loss += loss.item()
                    
                    predictions = torch.argmax(outputs.logits, dim=1)
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)
                    
                    val_predictions.extend(predictions.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            train_acc = train_correct / train_total if train_total > 0 else 0
            val_acc = val_correct / val_total if val_total > 0 else 0
            
            # Calculate precision, recall, F1
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_labels, val_predictions, average='binary', zero_division=0
            )
            
            print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")
            print(f"Val Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model("best_ranking_model_full.pt")
                print(f"Saved best model (Val Acc: {val_acc:.4f})")
            
            # Update learning rate
            scheduler.step()
            
            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer
        }, path)
        print(f"Model saved to {path}")

def main():
    """Main function for Step 3: Fine-tuning ranking model on full dataset"""
    print("Step 3: Fine-tuning Ranking LLM (Full Dataset)")
    print("=" * 60)
    
    # Initialize trainer
    trainer = RankingTrainer()
    
    # Prepare training data
    candidates_file = "step2_candidates_full.jsonl"  # Full dataset file
    
    try:
        # Use all data, no sampling
        training_examples = trainer.prepare_training_data(candidates_file, max_examples=None)
        
        if len(training_examples) < 100:
            print("Not enough training examples. Please run Step 2 first.")
            return
        
        # Train the model with optimized parameters
        trainer.train(training_examples, batch_size=16, epochs=10)
        
        print(f"\nStep 3 completed successfully!")
        print(f"Model saved as: best_ranking_model_full.pt")
        print(f"Ready for Step 4: Evaluation")
        
    except FileNotFoundError:
        print(f"Input file {candidates_file} not found. Please run Step 2 first.")
    except Exception as e:
        print(f"Error in Step 3: {e}")

if __name__ == "__main__":
    main() 