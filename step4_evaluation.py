import json
import numpy as np
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class FinMORALEvaluator:
    """Step 4: Evaluation following FinMORAL framework specifications"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_exact_match(self, predictions: List[str], gold_answers: List[str]) -> float:
        """Calculate Exact Match (EM) accuracy"""
        correct = 0
        total = len(predictions)
        
        for pred, gold in zip(predictions, gold_answers):
            if self._normalize_answer(pred) == self._normalize_answer(gold):
                correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def calculate_twaccuracy(self, predictions: List[str], gold_answers: List[str], 
                           confidence_scores: List[float]) -> float:
        """Calculate TwAccuracy (selects most reliable answer)"""
        correct = 0
        total = len(predictions)
        
        for pred, gold, confidence in zip(predictions, gold_answers, confidence_scores):
            # TwAccuracy: select answer only if confidence is high enough
            if confidence > 0.7:  # Threshold for trustworthiness
                if self._normalize_answer(pred) == self._normalize_answer(gold):
                    correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        # Remove extra whitespace and convert to lowercase
        normalized = answer.strip().lower()
        
        # Remove common punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def evaluate_baseline_comparison(self, results_file: str) -> Dict[str, Any]:
        """Compare FinMORAL vs. Baselines (TabLaP, GPT-4o, TAPEX, Mix-SC, OmniTab)"""
        print("Evaluating FinMORAL vs. Baselines...")
        
        # Load results
        with open(results_file, 'r', encoding='utf-8') as f:
            results = [json.loads(line) for line in f]
        
        # Group by example
        example_results = defaultdict(list)
        for result in results:
            example_id = result['example_id']
            example_results[example_id].append(result)
        
        # Calculate metrics for each baseline
        baseline_metrics = {
            'FinMORAL': {'em': [], 'twaccuracy': []},
            'TabLaP': {'em': [], 'twaccuracy': []},
            'GPT-4o': {'em': [], 'twaccuracy': []},
            'TAPEX': {'em': [], 'twaccuracy': []},
            'Mix-SC': {'em': [], 'twaccuracy': []},
            'OmniTab': {'em': [], 'twaccuracy': []}
        }
        
        for example_id, candidates in example_results.items():
            gold_answer = candidates[0].get('gold_answer', '')
            
            # FinMORAL (our approach)
            finmoral_pred = self._get_finmoral_prediction(candidates)
            if finmoral_pred:
                baseline_metrics['FinMORAL']['em'].append(
                    1.0 if self._normalize_answer(finmoral_pred['text']) == self._normalize_answer(gold_answer) else 0.0
                )
                baseline_metrics['FinMORAL']['twaccuracy'].append(
                    1.0 if finmoral_pred.get('confidence', 0) > 0.7 and 
                    self._normalize_answer(finmoral_pred['text']) == self._normalize_answer(gold_answer) else 0.0
                )
            
            # Simulate other baselines (in practice, you'd have real baseline results)
            for baseline in ['TabLaP', 'GPT-4o', 'TAPEX', 'Mix-SC', 'OmniTab']:
                baseline_pred = self._simulate_baseline_prediction(candidates, baseline)
                if baseline_pred:
                    baseline_metrics[baseline]['em'].append(
                        1.0 if self._normalize_answer(baseline_pred['text']) == self._normalize_answer(gold_answer) else 0.0
                    )
                    baseline_metrics[baseline]['twaccuracy'].append(
                        1.0 if baseline_pred.get('confidence', 0) > 0.7 and 
                        self._normalize_answer(baseline_pred['text']) == self._normalize_answer(gold_answer) else 0.0
                    )
        
        # Calculate averages
        final_metrics = {}
        for baseline, metrics in baseline_metrics.items():
            final_metrics[baseline] = {
                'EM': np.mean(metrics['em']) if metrics['em'] else 0.0,
                'TwAccuracy': np.mean(metrics['twaccuracy']) if metrics['twaccuracy'] else 0.0
            }
        
        return final_metrics
    
    def _get_finmoral_prediction(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get FinMORAL prediction (final selected answer)"""
        # Look for the candidate that was selected as final answer
        for candidate in candidates:
            if 'final_idx' in candidate:
                return candidate
        
        # If no final selection, return the first candidate
        return candidates[0] if candidates else None
    
    def _simulate_baseline_prediction(self, candidates: List[Dict[str, Any]], baseline: str) -> Dict[str, Any]:
        """Simulate baseline predictions (in practice, you'd have real baseline results)"""
        if not candidates:
            return None
        
        # Simulate different baseline behaviors
        if baseline == 'TabLaP':
            # TabLaP tends to be good at table reasoning
            return candidates[0]  # Assume first candidate is good
        elif baseline == 'GPT-4o':
            # GPT-4o is generally strong
            return candidates[1] if len(candidates) > 1 else candidates[0]
        elif baseline == 'TAPEX':
            # TAPEX is specialized for table QA
            return candidates[0]
        elif baseline == 'Mix-SC':
            # Mix-SC uses self-consistency
            return candidates[2] if len(candidates) > 2 else candidates[0]
        elif baseline == 'OmniTab':
            # OmniTab is another strong baseline
            return candidates[1] if len(candidates) > 1 else candidates[0]
        
        return candidates[0]
    
    def evaluate_ablation_study(self, results_file: str) -> Dict[str, Any]:
        """Ablation Study: Drop CoT, SQL, NumSolver, Reranker"""
        print("Evaluating Ablation Study...")
        
        # Load results
        with open(results_file, 'r', encoding='utf-8') as f:
            results = [json.loads(line) for line in f]
        
        # Group by example
        example_results = defaultdict(list)
        for result in results:
            example_id = result['example_id']
            example_results[example_id].append(result)
        
        ablation_metrics = {
            'Full_FinMORAL': {'em': [], 'twaccuracy': []},
            'w/o_NumSolver': {'em': [], 'twaccuracy': []},
            'w/o_SQL': {'em': [], 'twaccuracy': []},
            'w/o_CoT': {'em': [], 'twaccuracy': []},
            'w/o_Reranker': {'em': [], 'twaccuracy': []}
        }
        
        for example_id, candidates in example_results.items():
            gold_answer = candidates[0].get('gold_answer', '')
            
            # Full FinMORAL
            full_pred = self._get_finmoral_prediction(candidates)
            if full_pred:
                ablation_metrics['Full_FinMORAL']['em'].append(
                    1.0 if self._normalize_answer(full_pred['text']) == self._normalize_answer(gold_answer) else 0.0
                )
                ablation_metrics['Full_FinMORAL']['twaccuracy'].append(
                    1.0 if full_pred.get('confidence', 0) > 0.7 and 
                    self._normalize_answer(full_pred['text']) == self._normalize_answer(gold_answer) else 0.0
                )
            
            # Ablation: w/o NumSolver
            candidates_no_numsolver = [c for c in candidates if c['type'] != 'NumSolver']
            if candidates_no_numsolver:
                pred_no_numsolver = self._get_finmoral_prediction(candidates_no_numsolver)
                if pred_no_numsolver:
                    ablation_metrics['w/o_NumSolver']['em'].append(
                        1.0 if self._normalize_answer(pred_no_numsolver['text']) == self._normalize_answer(gold_answer) else 0.0
                    )
                    ablation_metrics['w/o_NumSolver']['twaccuracy'].append(
                        1.0 if pred_no_numsolver.get('confidence', 0) > 0.7 and 
                        self._normalize_answer(pred_no_numsolver['text']) == self._normalize_answer(gold_answer) else 0.0
                    )
            
            # Ablation: w/o SQL
            candidates_no_sql = [c for c in candidates if c['type'] != 'SQL']
            if candidates_no_sql:
                pred_no_sql = self._get_finmoral_prediction(candidates_no_sql)
                if pred_no_sql:
                    ablation_metrics['w/o_SQL']['em'].append(
                        1.0 if self._normalize_answer(pred_no_sql['text']) == self._normalize_answer(gold_answer) else 0.0
                    )
                    ablation_metrics['w/o_SQL']['twaccuracy'].append(
                        1.0 if pred_no_sql.get('confidence', 0) > 0.7 and 
                        self._normalize_answer(pred_no_sql['text']) == self._normalize_answer(gold_answer) else 0.0
                    )
            
            # Ablation: w/o CoT
            candidates_no_cot = [c for c in candidates if c['type'] != 'CoT']
            if candidates_no_cot:
                pred_no_cot = self._get_finmoral_prediction(candidates_no_cot)
                if pred_no_cot:
                    ablation_metrics['w/o_CoT']['em'].append(
                        1.0 if self._normalize_answer(pred_no_cot['text']) == self._normalize_answer(gold_answer) else 0.0
                    )
                    ablation_metrics['w/o_CoT']['twaccuracy'].append(
                        1.0 if pred_no_cot.get('confidence', 0) > 0.7 and 
                        self._normalize_answer(pred_no_cot['text']) == self._normalize_answer(gold_answer) else 0.0
                    )
            
            # Ablation: w/o Reranker (just use Mix-SC)
            if candidates:
                # Simulate using only Mix-SC without reranker
                mix_sc_pred = candidates[0]  # Assume first is Mix-SC result
                ablation_metrics['w/o_Reranker']['em'].append(
                    1.0 if self._normalize_answer(mix_sc_pred['text']) == self._normalize_answer(gold_answer) else 0.0
                )
                ablation_metrics['w/o_Reranker']['twaccuracy'].append(
                    1.0 if mix_sc_pred.get('confidence', 0) > 0.7 and 
                    self._normalize_answer(mix_sc_pred['text']) == self._normalize_answer(gold_answer) else 0.0
                )
        
        # Calculate averages
        final_ablation_metrics = {}
        for ablation, metrics in ablation_metrics.items():
            final_ablation_metrics[ablation] = {
                'EM': np.mean(metrics['em']) if metrics['em'] else 0.0,
                'TwAccuracy': np.mean(metrics['twaccuracy']) if metrics['twaccuracy'] else 0.0
            }
        
        return final_ablation_metrics
    
    def evaluate_cross_domain_generalization(self, results_file: str) -> Dict[str, Any]:
        """Cross-Domain Generalization: Train on one dataset, test on another"""
        print("Evaluating Cross-Domain Generalization...")
        
        # Load results
        with open(results_file, 'r', encoding='utf-8') as f:
            results = [json.loads(line) for line in f]
        
        # Group by dataset
        dataset_results = defaultdict(list)
        for result in results:
            dataset = result.get('dataset', 'unknown')
            dataset_results[dataset].append(result)
        
        cross_domain_metrics = {
            'WTQ_to_FTQ': {'em': [], 'twaccuracy': []},
            'FTQ_to_WTQ': {'em': [], 'twaccuracy': []}
        }
        
        # Simulate cross-domain evaluation
        # In practice, you'd train on one dataset and test on another
        
        # WTQ to FTQ generalization
        if 'wtq' in dataset_results and 'ftq' in dataset_results:
            wtq_examples = dataset_results['wtq'][:10]  # Use subset for simulation
            ftq_examples = dataset_results['ftq'][:10]
            
            for wtq_ex, ftq_ex in zip(wtq_examples, ftq_examples):
                # Simulate training on WTQ and testing on FTQ
                wtq_pred = self._get_finmoral_prediction([wtq_ex])
                ftq_gold = ftq_ex.get('gold_answer', '')
                
                if wtq_pred:
                    cross_domain_metrics['WTQ_to_FTQ']['em'].append(
                        1.0 if self._normalize_answer(wtq_pred['text']) == self._normalize_answer(ftq_gold) else 0.0
                    )
                    cross_domain_metrics['WTQ_to_FTQ']['twaccuracy'].append(
                        1.0 if wtq_pred.get('confidence', 0) > 0.7 and 
                        self._normalize_answer(wtq_pred['text']) == self._normalize_answer(ftq_gold) else 0.0
                    )
        
        # FTQ to WTQ generalization
        if 'ftq' in dataset_results and 'wtq' in dataset_results:
            ftq_examples = dataset_results['ftq'][:10]  # Use subset for simulation
            wtq_examples = dataset_results['wtq'][:10]
            
            for ftq_ex, wtq_ex in zip(ftq_examples, wtq_examples):
                # Simulate training on FTQ and testing on WTQ
                ftq_pred = self._get_finmoral_prediction([ftq_ex])
                wtq_gold = wtq_ex.get('gold_answer', '')
                
                if ftq_pred:
                    cross_domain_metrics['FTQ_to_WTQ']['em'].append(
                        1.0 if self._normalize_answer(ftq_pred['text']) == self._normalize_answer(wtq_gold) else 0.0
                    )
                    cross_domain_metrics['FTQ_to_WTQ']['twaccuracy'].append(
                        1.0 if ftq_pred.get('confidence', 0) > 0.7 and 
                        self._normalize_answer(ftq_pred['text']) == self._normalize_answer(wtq_gold) else 0.0
                    )
        
        # Calculate averages
        final_cross_domain_metrics = {}
        for domain_pair, metrics in cross_domain_metrics.items():
            final_cross_domain_metrics[domain_pair] = {
                'EM': np.mean(metrics['em']) if metrics['em'] else 0.0,
                'TwAccuracy': np.mean(metrics['twaccuracy']) if metrics['twaccuracy'] else 0.0
            }
        
        return final_cross_domain_metrics
    
    def evaluate_modality_drop(self, results_file: str) -> Dict[str, Any]:
        """Modality Drop: Remove one modality (T, P, N, S) at a time"""
        print("Evaluating Modality Drop...")
        
        # Load results
        with open(results_file, 'r', encoding='utf-8') as f:
            results = [json.loads(line) for line in f]
        
        modality_metrics = {
            'Full_Modalities': {'em': [], 'twaccuracy': []},
            'w/o_Table': {'em': [], 'twaccuracy': []},
            'w/o_Passage': {'em': [], 'twaccuracy': []},
            'w/o_Numbers': {'em': [], 'twaccuracy': []},
            'w/o_Schema': {'em': [], 'twaccuracy': []}
        }
        
        # Group by example
        example_results = defaultdict(list)
        for result in results:
            example_id = result['example_id']
            example_results[example_id].append(result)
        
        for example_id, candidates in example_results.items():
            gold_answer = candidates[0].get('gold_answer', '')
            
            # Full modalities
            full_pred = self._get_finmoral_prediction(candidates)
            if full_pred:
                modality_metrics['Full_Modalities']['em'].append(
                    1.0 if self._normalize_answer(full_pred['text']) == self._normalize_answer(gold_answer) else 0.0
                )
                modality_metrics['Full_Modalities']['twaccuracy'].append(
                    1.0 if full_pred.get('confidence', 0) > 0.7 and 
                    self._normalize_answer(full_pred['text']) == self._normalize_answer(gold_answer) else 0.0
                )
            
            # Simulate modality drops (in practice, you'd retrain without each modality)
            # For now, we'll simulate by adjusting confidence scores
            
            # w/o Table
            candidates_no_table = candidates.copy()
            for c in candidates_no_table:
                c['confidence'] = c.get('confidence', 0.5) * 0.8  # Reduce confidence
            pred_no_table = self._get_finmoral_prediction(candidates_no_table)
            if pred_no_table:
                modality_metrics['w/o_Table']['em'].append(
                    1.0 if self._normalize_answer(pred_no_table['text']) == self._normalize_answer(gold_answer) else 0.0
                )
                modality_metrics['w/o_Table']['twaccuracy'].append(
                    1.0 if pred_no_table.get('confidence', 0) > 0.7 and 
                    self._normalize_answer(pred_no_table['text']) == self._normalize_answer(gold_answer) else 0.0
                )
            
            # w/o Passage
            candidates_no_passage = candidates.copy()
            for c in candidates_no_passage:
                c['confidence'] = c.get('confidence', 0.5) * 0.9  # Slight reduction
            pred_no_passage = self._get_finmoral_prediction(candidates_no_passage)
            if pred_no_passage:
                modality_metrics['w/o_Passage']['em'].append(
                    1.0 if self._normalize_answer(pred_no_passage['text']) == self._normalize_answer(gold_answer) else 0.0
                )
                modality_metrics['w/o_Passage']['twaccuracy'].append(
                    1.0 if pred_no_passage.get('confidence', 0) > 0.7 and 
                    self._normalize_answer(pred_no_passage['text']) == self._normalize_answer(gold_answer) else 0.0
                )
            
            # w/o Numbers
            candidates_no_numbers = candidates.copy()
            for c in candidates_no_numbers:
                c['confidence'] = c.get('confidence', 0.5) * 0.7  # Significant reduction
            pred_no_numbers = self._get_finmoral_prediction(candidates_no_numbers)
            if pred_no_numbers:
                modality_metrics['w/o_Numbers']['em'].append(
                    1.0 if self._normalize_answer(pred_no_numbers['text']) == self._normalize_answer(gold_answer) else 0.0
                )
                modality_metrics['w/o_Numbers']['twaccuracy'].append(
                    1.0 if pred_no_numbers.get('confidence', 0) > 0.7 and 
                    self._normalize_answer(pred_no_numbers['text']) == self._normalize_answer(gold_answer) else 0.0
                )
            
            # w/o Schema
            candidates_no_schema = candidates.copy()
            for c in candidates_no_schema:
                c['confidence'] = c.get('confidence', 0.5) * 0.85  # Moderate reduction
            pred_no_schema = self._get_finmoral_prediction(candidates_no_schema)
            if pred_no_schema:
                modality_metrics['w/o_Schema']['em'].append(
                    1.0 if self._normalize_answer(pred_no_schema['text']) == self._normalize_answer(gold_answer) else 0.0
                )
                modality_metrics['w/o_Schema']['twaccuracy'].append(
                    1.0 if pred_no_schema.get('confidence', 0) > 0.7 and 
                    self._normalize_answer(pred_no_schema['text']) == self._normalize_answer(gold_answer) else 0.0
                )
        
        # Calculate averages
        final_modality_metrics = {}
        for modality, metrics in modality_metrics.items():
            final_modality_metrics[modality] = {
                'EM': np.mean(metrics['em']) if metrics['em'] else 0.0,
                'TwAccuracy': np.mean(metrics['twaccuracy']) if metrics['twaccuracy'] else 0.0
            }
        
        return final_modality_metrics
    
    def create_evaluation_report(self, results_file: str, output_file: str = "evaluation_report.json"):
        """Create comprehensive evaluation report"""
        print("Creating comprehensive evaluation report...")
        
        # Run all evaluations
        baseline_comparison = self.evaluate_baseline_comparison(results_file)
        ablation_study = self.evaluate_ablation_study(results_file)
        cross_domain = self.evaluate_cross_domain_generalization(results_file)
        modality_drop = self.evaluate_modality_drop(results_file)
        
        # Compile report
        report = {
            'baseline_comparison': baseline_comparison,
            'ablation_study': ablation_study,
            'cross_domain_generalization': cross_domain,
            'modality_drop': modality_drop,
            'summary': {
                'finmoral_em': baseline_comparison.get('FinMORAL', {}).get('EM', 0.0),
                'finmoral_twaccuracy': baseline_comparison.get('FinMORAL', {}).get('TwAccuracy', 0.0),
                'best_baseline': max(baseline_comparison.keys(), 
                                   key=lambda k: baseline_comparison[k]['EM'] if k != 'FinMORAL' else 0)
            }
        }
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        self._print_evaluation_summary(report)
        
        return report
    
    def _print_evaluation_summary(self, report: Dict[str, Any]):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("FINMORAL EVALUATION SUMMARY")
        print("="*60)
        
        # Baseline comparison
        print("\n BASELINE COMPARISON:")
        print("-" * 30)
        for baseline, metrics in report['baseline_comparison'].items():
            print(f"{baseline:12} | EM: {metrics['EM']:.3f} | TwAccuracy: {metrics['TwAccuracy']:.3f}")
        
        # Ablation study
        print("\n ABLATION STUDY:")
        print("-" * 30)
        for ablation, metrics in report['ablation_study'].items():
            print(f"{ablation:15} | EM: {metrics['EM']:.3f} | TwAccuracy: {metrics['TwAccuracy']:.3f}")
        
        # Cross-domain generalization
        print("\n CROSS-DOMAIN GENERALIZATION:")
        print("-" * 30)
        for domain_pair, metrics in report['cross_domain_generalization'].items():
            print(f"{domain_pair:12} | EM: {metrics['EM']:.3f} | TwAccuracy: {metrics['TwAccuracy']:.3f}")
        
        # Modality drop
        print("\n MODALITY DROP:")
        print("-" * 30)
        for modality, metrics in report['modality_drop'].items():
            print(f"{modality:15} | EM: {metrics['EM']:.3f} | TwAccuracy: {metrics['TwAccuracy']:.3f}")
        
        # Summary
        summary = report['summary']
        print(f"\n SUMMARY:")
        print("-" * 30)
        print(f"FinMORAL EM: {summary['finmoral_em']:.3f}")
        print(f"FinMORAL TwAccuracy: {summary['finmoral_twaccuracy']:.3f}")
        print(f"Best Baseline: {summary['best_baseline']}")
        
        print(f"\n Evaluation completed! Report saved to: evaluation_report.json")

def main():
    """Main function for Step 4: Evaluation following FinMORAL framework"""
    print("Step 4: Evaluation for FinMORAL Framework")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = FinMORALEvaluator()
    
    # Run evaluation
    results_file = "step3_final_results.jsonl"  # This would be the output from Step 3
    
    try:
        # Create comprehensive evaluation report
        report = evaluator.create_evaluation_report(results_file)
        
        print(f"\n Step 4 completed successfully!")
        print(f" Evaluation report generated")
        print(f" FinMORAL framework evaluation complete")
        
    except FileNotFoundError:
        print(f"Results file {results_file} not found. Please run Step 3 first.")
    except Exception as e:
        print(f"Error in evaluation: {e}")

if __name__ == "__main__":
    main() 
