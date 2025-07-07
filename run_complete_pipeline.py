#!/usr/bin/env python3
"""
FinMORAL Framework Complete Pipeline
====================================

This script runs the complete FinMORAL framework pipeline:
1. Dataset Preparation (WTQ + FTQ)
2. Candidate Generation (SQL + NumSolver + CoT)
3. Final Answer Selection (Mix-SC + DistilBERT Reranker)
4. Evaluation (EM + TwAccuracy + Ablations)

Based on the FinMORAL research paper framework.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import subprocess

class FinMORALPipeline:
    """Complete FinMORAL framework pipeline orchestrator"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.step_outputs = {
            'step1': 'step1_processed_data.jsonl',
            'step2': 'step2_candidates.jsonl',
            'step3': 'step3_final_results.jsonl',
            'step4': 'evaluation_report.json'
        }
        
        # Pipeline configuration
        self.config = {
            'max_examples': 100,  # Limit for testing
            'use_mock_apis': True,  # Set to False for real API calls
            'save_intermediate': True,
            'evaluate_ablation': True,
            'evaluate_cross_domain': True
        }
    
    def run_step1_data_preparation(self) -> bool:
        """Run Step 1: Dataset Preparation for FinMORAL framework"""
        print("\n" + "="*60)
        print("STEP 1: Dataset Preparation (FinMORAL Framework)")
        print("="*60)
        print("• Loading WTQ and FTQ datasets")
        print("• Extracting q, T, P, N, S components")
        print("• Creating FinMORAL-compatible format")
        
        try:
            # Import and run Step 1
            from step1_data_preparation import main as step1_main
            step1_main()
            
            # Verify output
            if os.path.exists(self.step_outputs['step1']):
                with open(self.step_outputs['step1'], 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                print(f" Step 1 completed: {len(lines)} examples processed")
                return True
            else:
                print(" Step 1 failed: Output file not found")
                return False
                
        except Exception as e:
            print(f" Step 1 failed: {e}")
            return False
    
    def run_step2_candidate_generation(self) -> bool:
        """Run Step 2: Candidate Generation using FinMORAL modules"""
        print("\n" + "="*60)
        print("STEP 2: Candidate Answer Generation (FinMORAL Framework)")
        print("="*60)
        print("• SQL Module: Generate and execute SQL queries")
        print("• NumSolver: Symbolic arithmetic reasoning")
        print("• CoT Reasoning: Step-by-step with self-consistency (k=5)")
        
        try:
            # Import and run Step 2
            from step2_candidate_generation import main as step2_main
            step2_main()
            
            # Verify output
            if os.path.exists(self.step_outputs['step2']):
                with open(self.step_outputs['step2'], 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                print(f" Step 2 completed: {len(lines)} candidates generated")
                return True
            else:
                print(" Step 2 failed: Output file not found")
                return False
                
        except Exception as e:
            print(f" Step 2 failed: {e}")
            return False
    
    def run_step3_final_selection(self) -> bool:
        """Run Step 3: Final Answer Selection (Mix-SC + DistilBERT Reranker)"""
        print("\n" + "="*60)
        print("STEP 3: Final Answer Selection (FinMORAL Framework)")
        print("="*60)
        print("• Mix-SC Self-Consistency Voting")
        print("• DistilBERT Pairwise Reranker")
        print("• Combined selection strategy")
        
        try:
            # Import and run Step 3
            from step3_ranking_finetune import main as step3_main
            step3_main()
            
            # Verify output
            if os.path.exists(self.step_outputs['step3']):
                with open(self.step_outputs['step3'], 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                print(f" Step 3 completed: {len(lines)} final answers selected")
                return True
            else:
                print(" Step 3 failed: Output file not found")
                return False
                
        except Exception as e:
            print(f" Step 3 failed: {e}")
            return False
    
    def run_step4_evaluation(self) -> bool:
        """Run Step 4: Evaluation following FinMORAL specifications"""
        print("\n" + "="*60)
        print("STEP 4: Evaluation (FinMORAL Framework)")
        print("="*60)
        print("• EM (Exact Match) accuracy")
        print("• TwAccuracy (trustworthy answer selection)")
        print("• Ablation studies (w/o modules)")
        print("• Cross-domain generalization")
        print("• Modality drop analysis")
        
        try:
            # Import and run Step 4
            from step4_evaluation import main as step4_main
            step4_main()
            
            # Verify output
            if os.path.exists(self.step_outputs['step4']):
                with open(self.step_outputs['step4'], 'r', encoding='utf-8') as f:
                    report = json.load(f)
                print(f" Step 4 completed: Evaluation report generated")
                print(f"   FinMORAL EM: {report.get('summary', {}).get('finmoral_em', 0):.3f}")
                print(f"   FinMORAL TwAccuracy: {report.get('summary', {}).get('finmoral_twaccuracy', 0):.3f}")
                return True
            else:
                print(" Step 4 failed: Output file not found")
                return False
                
        except Exception as e:
            print(f" Step 4 failed: {e}")
            return False
    
    def create_final_report(self) -> Dict[str, Any]:
        """Create comprehensive final report"""
        print("\n" + "="*60)
        print("CREATING FINAL REPORT")
        print("="*60)
        
        report = {
            'framework': 'FinMORAL',
            'description': 'Financial Multi-modal Reasoning with Answer Selection',
            'steps_completed': [],
            'dataset_info': {},
            'model_info': {},
            'evaluation_results': {},
            'ablation_results': {},
            'cross_domain_results': {},
            'modality_drop_results': {},
            'summary': {}
        }
        
        # Check which steps completed
        for step, output_file in self.step_outputs.items():
            if os.path.exists(output_file):
                report['steps_completed'].append(step)
        
        # Load evaluation results if available
        if os.path.exists(self.step_outputs['step4']):
            with open(self.step_outputs['step4'], 'r', encoding='utf-8') as f:
                eval_report = json.load(f)
                report['evaluation_results'] = eval_report.get('baseline_comparison', {})
                report['ablation_results'] = eval_report.get('ablation_study', {})
                report['cross_domain_results'] = eval_report.get('cross_domain_generalization', {})
                report['modality_drop_results'] = eval_report.get('modality_drop', {})
                report['summary'] = eval_report.get('summary', {})
        
        # Add model information
        report['model_info'] = {
            'sql_module': 'Pointer-generator Transformer (4-layer, 512-dim)',
            'numsolver': 'Symbolic arithmetic parser (tree-based)',
            'cot_reasoning': 'GPT-4.5 with self-consistency (k=5)',
            'reranker': 'DistilBERT pairwise classifier',
            'voting': 'Mix-SC (consistency + heuristic)'
        }
        
        # Add dataset information
        report['dataset_info'] = {
            'wtq': 'WikiTableQuestions (SQL + arithmetic focus)',
            'ftq': 'Filtered FeTaQA (financial QA with natural language)',
            'total_examples': len(report['steps_completed']) * 100  # Estimate
        }
        
        # Save final report
        with open('finmoral_final_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f" Final report saved: finmoral_final_report.json")
        return report
    
    def print_pipeline_summary(self, report: Dict[str, Any]):
        """Print comprehensive pipeline summary"""
        print("\n" + "="*60)
        print("FINMORAL PIPELINE SUMMARY")
        print("="*60)
        
        print(f"Framework: {report['framework']}")
        print(f"Description: {report['description']}")
        print(f"Steps Completed: {len(report['steps_completed'])}/4")
        
        for step in report['steps_completed']:
            print(f"   {step}")
        
        if report['summary']:
            print(f"\n Key Results:")
            print(f"  FinMORAL EM: {report['summary'].get('finmoral_em', 0):.3f}")
            print(f"  FinMORAL TwAccuracy: {report['summary'].get('finmoral_twaccuracy', 0):.3f}")
            print(f"  Best Baseline: {report['summary'].get('best_baseline', 'N/A')}")
        
        print(f"\n Datasets Used:")
        for dataset, description in report['dataset_info'].items():
            print(f"  {dataset.upper()}: {description}")
        
        print(f"\n Models Used:")
        for model, description in report['model_info'].items():
            print(f"  {model}: {description}")
        
        print(f"\n Output Files:")
        for step, output_file in self.step_outputs.items():
            if step in report['steps_completed']:
                print(f"  {step}: {output_file}")
        
        print(f"\n FinMORAL pipeline execution completed!")
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete FinMORAL pipeline"""
        print(" Starting FinMORAL Framework Pipeline")
        print("="*60)
        print("This pipeline implements the FinMORAL framework with:")
        print("• 2 datasets: WTQ + FTQ")
        print("• 3 specialized modules: SQL + NumSolver + CoT")
        print("• 2 selection strategies: Mix-SC + DistilBERT Reranker")
        print("• Comprehensive evaluation: EM + TwAccuracy + Ablations")
        print("="*60)
        
        start_time = time.time()
        
        # Run each step
        steps = [
            ('Step 1: Data Preparation', self.run_step1_data_preparation),
            ('Step 2: Candidate Generation', self.run_step2_candidate_generation),
            ('Step 3: Final Selection', self.run_step3_final_selection),
            ('Step 4: Evaluation', self.run_step4_evaluation)
        ]
        
        completed_steps = []
        
        for step_name, step_function in steps:
            print(f"\n Running {step_name}...")
            if step_function():
                completed_steps.append(step_name)
                print(f" {step_name} completed successfully")
            else:
                print(f" {step_name} failed")
                break
        
        # Create final report
        if len(completed_steps) >= 3:  # At least 3 steps completed
            report = self.create_final_report()
            self.print_pipeline_summary(report)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"\n  Total execution time: {duration:.2f} seconds")
            print(f" Pipeline success rate: {len(completed_steps)}/4 steps")
            
            return True
        else:
            print(f"\n Pipeline failed: Only {len(completed_steps)}/4 steps completed")
            return False

def main():
    """Main function to run the complete FinMORAL pipeline"""
    # Check if required files exist
    required_files = [
        'step1_data_preparation.py',
        'step2_candidate_generation.py', 
        'step3_ranking_finetune.py',
        'step4_evaluation.py'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f" Missing required files: {missing_files}")
        print("Please ensure all step files are present in the current directory.")
        return False
    
    # Check if datasets exist
    dataset_paths = ['WTQdata', 'fetaQAdata']
    missing_datasets = [d for d in dataset_paths if not os.path.exists(d)]
    if missing_datasets:
        print(f" Missing datasets: {missing_datasets}")
        print("Please ensure WTQdata and fetaQAdata directories are present.")
        return False
    
    # Initialize and run pipeline
    pipeline = FinMORALPipeline()
    success = pipeline.run_complete_pipeline()
    
    if success:
        print(f"\n FinMORAL Framework Pipeline completed successfully!")
        print(f" Check the generated files for detailed results:")
        print(f"   • step1_processed_data.jsonl - Processed datasets")
        print(f"   • step2_candidates.jsonl - Generated candidates")
        print(f"   • step3_final_results.jsonl - Final answers")
        print(f"   • evaluation_report.json - Evaluation results")
        print(f"   • finmoral_final_report.json - Complete summary")
    else:
        print(f"\n Pipeline execution failed. Check the logs above for details.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 