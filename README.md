# Machine Learning Assignment 2: Bagging, Boosting, and Bias-Variance on Face Recognition

**Student Details:**
- **Name:** Ranvir Kumar Singh
- **Roll Number:** P25CS0007
- **Random Seed:** `250007` (derived deterministically by extracting all numeric digits of the roll number in order)
- **Course:** Machine Learning (Assignment 2)
- **Report Format:** CVPR 2-column format (`ML_Assignment_2_Report.pdf`, 6 pages)

---

## Executive Summary of Results

| Model | Clean Test Acc | Masked Test Acc (16x16 Occlusion) | Bias² (30 Refits) | Variance (30 Refits) | Total MSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Single Decision Tree Baseline** ($d=\text{None}, s=1$) | 58.75% | 30.00% (-28.75%) | 0.3888 | 0.6529 | 1.0417 |
| **Final Bagged Ensemble** ($B=100$) | **88.75%** | **77.50%** (-11.25%) | 0.5105 | **0.0253** (-96.1%) | **0.5358** |
| **Selected SAMME Boosting** ($d=4, T=100$) | 58.75% | 50.00% (-8.75%) | 0.8843 | **0.0184** | 0.9027 |

### Core Findings
1. **Dramatic Variance Elimination:** Bagging unpruned trees eliminates **96.1%** of estimation variance (dropping from 0.6529 to 0.0253), boosting clean test accuracy from 58.75% to 88.75% and masked test accuracy from 30.00% to 77.50%.
2. **Representational Bottleneck in Boosting:** SAMME AdaBoost with shallow trees exhibits severe bias on 40-class raw image data ($\text{Bias}^2 = 0.8843$). Weak learners misclassify atypical samples repeatedly (Face #55 misclassified in 88% of rounds, escalating weight by 12.9x), skewing decision boundaries.
3. **Critical Depth-Gain Discovery:** In the depth sweep ($d \in \{4, 16, \text{None}\}$), the largest test accuracy improvement from bagging occurs at **intermediate depth ($d=16$) with a +35.79% surge**, not at unconstrained depth (+28.79%). Balanced leaves capture real structural clusters, and bagging stabilizes their margin boundaries.

---

## Repository Structure

```
Ranveer_ML/
├── .gitignore
├── requirements.txt
├── README.md
├── ML_Assignment_2_Final-3.docx     # Assignment problem specification
├── ML_Assignment_2_Report.pdf       # Compiled CVPR 2-column PDF report
├── data/
│   ├── split.npz                    # Stratified train/val/test split (seed 250007)
│   ├── q1_results.json              # Decision tree grid search & Bagging diagnostics
│   ├── q2_results.json              # Boosting grid search & dynamics logs
│   ├── q2_sample_weights.npy        # Boosting sample weights trajectory
│   └── q3_results.json              # 30-refit Bias-Variance decomposition logs
├── src/
│   ├── __init__.py
│   ├── dataset.py                   # Olivetti dataset fetcher, stratified split & occlusion mask
│   ├── decision_tree.py             # Decision tree grid search & structural metrics
│   ├── bagging.py                   # Scratch Bagging implementation & diagnostics
│   ├── boosting.py                  # Scratch SAMME multiclass AdaBoost implementation
│   └── bias_variance.py             # 30-refit empirical bias-variance estimator
├── experiments/
│   ├── __init__.py
│   ├── run_q1.py                    # Question 1 experiment runner & figure generator
│   ├── run_q2.py                    # Question 2 experiment runner & figure generator
│   ├── run_q3.py                    # Question 3 experiment runner & figure generator
│   └── run_all.py                   # Master runner (runs all experiments deterministically)
├── figures/
│   ├── fig1_q1c_diagnostics.png     # Bagging OOB accuracy, disagreement & vote margin
│   ├── fig2_q1d_face_analysis.png   # Case study of face predictions & top-5 votes
│   ├── fig3_q2b_boosting_dynamics.png # Boosting training error, alpha, and weight concentration
│   ├── fig4_q2c_heavy_sample.png    # Heaviest training face inspection (Face #55, Class 9)
│   ├── fig5_q2d_masked_diff_face.png# Differential response on masked test face (Face #0)
│   └── fig6_q3b_bias_variance.png   # Bias-variance decomposition across tree depths
└── reports/
    └── ML_Assignment_2_Report.typ   # Typst source code for CVPR report
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.9+ (Python 3.10 / 3.11 / 3.12 recommended)
- `pip`
- [Typst](https://github.com/typst/typst) (optional, only needed if modifying and recompiling the PDF report)

### 2. Environment Setup
```bash
# Clone the repository (or navigate to workspace directory)
cd /path/to/Ranveer_ML

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required Python packages
pip install -r requirements.txt
```

---

## Running the Code & Reproducing Results

All random seeds are locked to `250007`. Every experiment will yield identical results to those reported in the PDF.

### Run All Experiments (End-to-End)
To execute the complete pipeline across all three questions and generate all figures and logs:
```bash
python3 -m experiments.run_all
```

### Run Individual Experiments
You can also run each question independently:
```bash
# Question 1: Decision Tree Grid Search & Bagging from Scratch
python3 -m experiments.run_q1

# Question 2: SAMME AdaBoost from Scratch & Dynamic Analysis
python3 -m experiments.run_q2

# Question 3: 30-Repetition Bias-Variance Decomposition
python3 -m experiments.run_q3
```

---

## Compiling the Report

The CVPR-style report is authored in [Typst](https://typst.app), a modern academic document typesetting system that compiles directly to PDF:

```bash
# Recompile the PDF report from workspace root
typst compile --root . reports/ML_Assignment_2_Report.typ ML_Assignment_2_Report.pdf
```

The resulting `ML_Assignment_2_Report.pdf` conforms strictly to:
- US Letter paper, 0.75-inch margins
- Two-column format
- Times New Roman typography
- Clear section headings and academic references
- Exactly 6 pages (well within the maximum 8-page limit)

---

## Pushing to GitHub (Instructions for Student)

When you are ready to upload this codebase to your GitHub repository:

1. Create a new repository on [GitHub](https://github.com/new) (e.g. `ML-Assignment-2`).
2. Run the following terminal commands:
   ```bash
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git branch -M main
   git push -u origin main
   ```
3. If you wish to update the GitHub URL or commit hash displayed on page 1 of the report:
   - Edit the metadata block in `reports/ML_Assignment_2_Report.typ`.
   - Recompile: `typst compile --root . reports/ML_Assignment_2_Report.typ ML_Assignment_2_Report.pdf`.
   - Commit and push:
     ```bash
     git commit -am "Update GitHub URL and commit hash in report"
     git push
     ```

---

## Submission Checklist

- [x] Deterministic seed (`250007`) derived from roll number `P25CS0007`.
- [x] Stratified split into 6 train, 2 validation, 2 test images per identity.
- [x] Centered $16 \times 16$ pixel occlusion mask generated and evaluated.
- [x] Scikit-learn Decision Tree baseline hyperparameter exploration ($d \in \{2, 4, 8, 16, \text{None}\}$, $s \in \{1, 2, 4\}$).
- [x] Decision tree selection strictly on validation accuracy ($d=\text{None}, s=1$).
- [x] Scratch Bootstrap Aggregation (Bagging) implementation with theoretical expectation derivation ($151.89$ distinct samples).
- [x] Out-of-bag accuracy, pairwise tree disagreement, and mean vote margin evaluated across $B \in \{1, 5, 10, 25, 50, 100, 200\}$.
- [x] Clean vs. masked robustness benchmarked with concrete error taxonomy case studies.
- [x] Scratch Multiclass SAMME AdaBoost implementation with random-guessing threshold safeguard.
- [x] Toy 3-class ($N=6$) analytical verification logged.
- [x] Boosting grid search over weak learner depths and boosting rounds.
- [x] Round-by-round training dynamics logged (weighted error, learner weight, sample weights).
- [x] Heaviest training sample (#55, Class 9) inspected and analyzed.
- [x] Occlusion differential analysis between Bagging and Boosting (Face #0).
- [x] 30-repetition empirical bias-variance decomposition protocol implemented without synthetic assumptions.
- [x] Exact vector MSE identity validated ($\text{Total MSE} = \text{Bias}^2 + \text{Variance}$).
- [x] Depth sweep decomposition across shallow ($d=4$), deep ($d=16$), and unpruned ($d=\text{None}$).
- [x] Verified and justified why intermediate depth ($d=16$) achieves the maximal test accuracy gain (+35.79%).
- [x] Master synthesis discussion addressing all three core assignment questions.
- [x] 6-page CVPR two-column PDF generated (`ML_Assignment_2_Report.pdf`).
- [x] Git repository initialized and all code committed.
