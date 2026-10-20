"""
Master execution script for ML Assignment 2: Bagging, Boosting, and Bias-Variance on Face Recognition.
Student Name: Ranvir Kumar Singh
Roll Number: P25CS0007
Seed: 250007
"""

import sys
import os
import subprocess
import time

def run_master():
    print("=" * 80)
    print("MACHINE LEARNING ASSIGNMENT 2: END-TO-END MASTER REPRODUCTION PIPELINE")
    print("Student Name: Ranvir Kumar Singh | Roll Number: P25CS0007 | Seed: 250007")
    print("=" * 80)
    
    t_start = time.time()
    
    # Step 1: Dataset preparation
    print("\n[STEP 1/5] Preparing Olivetti Faces Dataset & Stratified Splits...")
    from src.dataset import prepare_dataset
    prepare_dataset()
    
    # Step 2: Question 1
    print("\n[STEP 2/5] Running Question 1: Decision Tree Baseline & Bagging from Scratch...")
    from experiments.run_q1 import run_q1_experiments
    run_q1_experiments()
    
    # Step 3: Question 2
    print("\n[STEP 3/5] Running Question 2: Multiclass Boosting from Scratch (SAMME)...")
    from experiments.run_q2 import run_q2_experiments
    run_q2_experiments()
    
    # Step 4: Question 3
    print("\n[STEP 4/5] Running Question 3: Bias and Variance as Measured Quantities...")
    from experiments.run_q3 import run_q3_experiments
    run_q3_experiments()
    
    # Step 5: Compile CVPR Report
    print("\n[STEP 5/5] Compiling CVPR Two-Column 8-Page PDF Report...")
    res = subprocess.run(["typst", "compile", "reports/ML_Assignment_2_Report.typ", "ML_Assignment_2_Report.pdf"], capture_output=True, text=True)
    if res.returncode == 0:
        print("Successfully compiled PDF: ML_Assignment_2_Report.pdf")
    else:
        print("Typst compile output:", res.stdout, res.stderr)
        
    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETE IN {total_time:.1f} SECONDS!")
    print("All deliverables, figures, logs, and report PDF are generated and ready for submission.")
    print("=" * 80)

if __name__ == "__main__":
    run_master()
