#set page(
  paper: "us-letter",
  margin: (x: 0.75in, top: 0.75in, bottom: 0.85in),
  header: align(right)[
    #text(8pt, fill: luma(100))[
      CVPR 2026 Coursework --- Machine Learning Assignment 2
    ]
  ],
  footer: align(center)[#context text(9pt)[#counter(page).display("1")]]
)

#set text(
  font: "Times New Roman",
  size: 9.5pt,
  spacing: 120%,
  lang: "en"
)

#set par(
  justify: true,
  leading: 0.55em,
  first-line-indent: 1.2em
)

#set heading(numbering: "1.1")

// Macro for section headings
#show heading.where(level: 1): it => {
  v(0.8em, weak: true)
  text(10pt, weight: "bold")[
    #counter(heading).display(it.numbering) #h(0.4em) #upper(it.body)
  ]
  v(0.4em, weak: true)
}

#show heading.where(level: 2): it => {
  v(0.6em, weak: true)
  text(9.5pt, weight: "bold")[
    #counter(heading).display(it.numbering) #h(0.4em) #it.body
  ]
  v(0.3em, weak: true)
}

#show heading.where(level: 3): it => {
  v(0.5em, weak: true)
  text(9.5pt, style: "italic")[
    #counter(heading).display(it.numbering) #h(0.4em) #it.body
  ]
  v(0.25em, weak: true)
}

// -------------------------------------------------------------
// TITLE & METADATA (Spans full page width)
// -------------------------------------------------------------
#align(center)[
  #v(0.5em)
  #text(16pt, weight: "bold")[
    Bagging, Boosting, and Bias-Variance Decomposition\
    on Face Recognition: An Empirical Study on Olivetti Faces
  ]
  #v(0.8em)
  
  #text(11pt, weight: "bold")[Ranvir Kumar Singh] \
  #text(10pt)[Roll Number: *P25CS0007*] \
  #text(9pt, style: "italic")[Department of Computer Science and Engineering] \
  #v(0.4em)
  #text(9pt)[
    *GitHub Repository:* #link("https://github.com/ranvir-kumar-singh/ML-Assignment-2")[https://github.com/ranvir-kumar-singh/ML-Assignment-2] \
    *Final Commit Hash:* `5c302ae5f09d05ace48d526326a371a69aef55e8`
  ]
  #v(1.0em)
]

// -------------------------------------------------------------
// ABSTRACT (Centered block)
// -------------------------------------------------------------
#align(center)[
  #block(width: 90%)[
    #text(9pt, weight: "bold")[Abstract] \
    #text(8.8pt, style: "italic")[
      Ensemble learning methods systematically manipulate the bias-variance trade-off to enhance generalization beyond individual hypothesis spaces. In this investigation, we conduct an extensive empirical study of Decision Trees, Bootstrap Aggregation (Bagging), and Multiclass AdaBoost (SAMME) on the Olivetti Faces dataset (40 identities, 4096 raw pixel dimensions). Using a deterministic, roll-number-derived seed (250007) and stratified partitions (6 train, 2 validation, 2 test images per identity), we implement Bagging and SAMME from scratch and rigorously evaluate their learning mechanics. We examine bootstrap sample uniqueness, diagnostic signals (out-of-bag accuracy, pairwise disagreement, vote margins), and training dynamics across varying weak-learner capacities. Robustness to localized structural corruption is benchmarked using a centered 16 by 16 pixel occlusion mask. Finally, we execute a 30-repetition empirical bias-variance decomposition in class probability space across depths. Our findings reveal that Bagging achieves dramatic variance reduction (from 0.6529 down to 0.0253, a 96.1% decrease), driving test accuracy from 58.75% to 88.75% on clean faces and from 30.00% to 77.50% on occluded faces. Crucially, we demonstrate that while unpruned trees experience the largest absolute variance removal, intermediate depths yield the greatest relative accuracy improvements, exposing the critical interplay between margin geometry, leaf occupancy, and ensemble diversity.
    ]
  ]
]

#v(1.0em)

// -------------------------------------------------------------
// TWO-COLUMN BODY
// -------------------------------------------------------------
#show: rest => columns(2, gutter: 0.24in, rest)

= Introduction & Experimental Protocol

Face recognition in raw high-dimensional image spaces presents fundamental statistical challenges: severe sample sparsity ($N << D$), localized illumination shifts, non-linear intra-class pose variations, and sensitivity to occlusions. While modern deep neural networks achieve near-perfect face verification, understanding the exact bias-variance trade-offs of classical non-parametric tree ensembles provides foundational insights into ensemble variance reduction and margin maximization.

#figure(
  image("../figures/fig1_q1c_diagnostics.png", width: 95%),
  caption: [Bagging diagnostic trajectories as a function of ensemble size $B$: Out-of-Bag (OOB) accuracy, average pairwise tree disagreement on validation data, and mean vote margin. Selected $B=100$ marked by the dotted line.]
) <fig1_diagnostics>

== Dataset & Stratified Partitioning

We utilize the Cambridge Olivetti Faces benchmark, comprising 40 distinct human subjects with 10 grayscale images each, captured at $64 times 64$ resolution. The input space consists solely of raw flattened pixel intensities $x in [0, 1]^(4096)$. No principal component analysis (PCA) or dimensionality reduction is applied, preserving the full high-dimensional geometry.

To ensure total reproducibility, all numeric characters of the complete roll number (*P25CS0007*) are extracted in order, yielding the unique pseudo-random seed:
$ "Seed" = 250007 $
Using `numpy.random.default_rng(250007)`, each identity's 10 images are partitioned into:
- *Training set:* 6 images per identity ($N_("train") = 240$, 60%),
- *Validation set:* 2 images per identity ($N_("val") = 80$, 20%),
- *Test set:* 2 images per identity ($N_("test") = 80$, 20%).

This split is serialized to disk (`data/split.npz`) and reused across all experiments.

== Occlusion Robustness Setup

To measure robustness against localized spatial perturbation without retraining, we construct an occluded test set by zeroing the centered $16 times 16$ block:
$ "Mask Region": "rows" 24 "to" 39, "cols" 24 "to" 39 $
This corrupts key facial features (nasal bridge, eyes, and central cheekbones) across all 80 test faces. All models are trained strictly on clean images.

= Question 1: Decision Tree Baseline & Bagging from Scratch

== Single Tree Baseline Exploration

We systematically train scikit-learn's `DecisionTreeClassifier` across a 2D hyperparameter grid: maximum depth $d in {2, 4, 8, 16, "None"}$ and minimum samples per leaf $s in {1, 2, 4}$. Model selection is conducted *strictly using validation set accuracy* without exposure to the test set.

#figure(
  table(
    columns: (1.2fr, 1.1fr, 1.1fr, 1.1fr, 1.1fr, 1fr, 1.1fr, 1.1fr),
    stroke: 0.5pt + luma(180),
    align: center,
    [#strong[Depth]], [#strong[MinLeaf]], [#strong[Train]], [#strong[Val]], [#strong[Test]], [#strong[Leaves]], [#strong[S/Leaf]], [#strong[Singl.]],
    [2], [1], [0.075], [0.075], [0.075], [3], [80.0], [0.000],
    [2], [2], [0.075], [0.075], [0.075], [3], [80.0], [0.000],
    [2], [4], [0.075], [0.075], [0.075], [3], [80.0], [0.000],
    [4], [1], [0.125], [0.113], [0.100], [5], [48.0], [0.000],
    [4], [2], [0.125], [0.113], [0.100], [5], [48.0], [0.000],
    [4], [4], [0.125], [0.113], [0.100], [5], [48.0], [0.000],
    [8], [1], [0.225], [0.188], [0.175], [9], [26.7], [0.000],
    [8], [2], [0.225], [0.188], [0.175], [9], [26.7], [0.000],
    [8], [4], [0.225], [0.188], [0.175], [9], [26.7], [0.000],
    [16], [1], [0.425], [0.325], [0.288], [17], [14.1], [0.000],
    [16], [2], [0.425], [0.325], [0.288], [17], [14.1], [0.000],
    [16], [4], [0.425], [0.325], [0.288], [17], [14.1], [0.000],
    [*None*], [*1*], [*1.000*], [*0.650*], [*0.588*], [*45*], [*5.33*], [*0.111*],
    [None], [2], [0.979], [0.638], [0.575], [45], [5.33], [0.000],
    [None], [4], [0.979], [0.613], [0.575], [40], [6.00], [0.000],
  ),
  caption: [Decision tree hyperparameter grid search on Olivetti Faces. Bold row denotes frozen baseline model selected by validation accuracy.]
) <tab_dt_grid>

*Structural Evolution across Depths:* As tabulated in @tab_dt_grid, depth fundamentally dictates model complexity:
1. *Constrained Depths ($d <= 8$):* At $d=2$, the tree creates only 3 leaves averaging 80 samples each. Because 40 identities must be resolved, the tree lacks the capacity to partition the space, yielding high bias (Val Acc: 7.5%, barely above random guessing at 2.5%).
2. *Intermediate Depths ($d=16$):* Tree splits expand to 17 leaves (14.1 samples/leaf), allowing partial separation of identity clusters (Val Acc: 32.5%).
3. *Unconstrained Depth ($d="None"$):* The tree grows to 45 leaves, achieving 100% training accuracy. The average leaf occupancy drops to 5.33 samples, and 11.11% of leaves become singletons (containing exactly one training instance). Here, individual axis-aligned hyperplanes isolate specific training photos. While training error vanishes, test accuracy lags at 58.75%, evidencing severe estimation variance.

*Selected Frozen Baseline:* The optimal validation accuracy (0.6500) corresponds to $d="None", s=1$. This configuration is frozen for all subsequent bagging experiments.

#figure(
  image("../figures/fig2_q1d_face_analysis.png", width: 100%),
  caption: [Error taxonomy case analysis: Clean test faces and top-5 voting distributions ($B=100$). Rows 1--2: Bagging corrects single tree misclassifications via consensus voting. Rows 3--4: Severe shared failure modes where both models misclassify.]
) <fig2_cases>

== Bagging from Scratch & Bootstrap Uniqueness

We implement Bagging entirely from scratch using bootstrap sampling ($N=240$ drawn uniformly with replacement) applied to the frozen baseline tree. We evaluate ensemble sizes $B in {1, 5, 10, 25, 50, 100, 200}$.

*Theoretical Bootstrap Expectation:* For a training set of size $N$, the probability that any specific sample $i$ is not selected in a single draw is $1 - 1/N$. After $N$ independent draws with replacement, the probability that sample $i$ is omitted is:
$ P(i in.not "Bootstrap") = (1 - 1/N)^N approx e^(-1) approx 0.36788 $
The expected number of distinct examples drawn is:
$ bb(E)[D] = N [1 - (1 - 1/N)^N] $
For $N = 240$:
$ bb(E)[D] = 240 [1 - (239/240)^(240)] = 151.8932 quad (63.29%) $

#figure(
  table(
    columns: (1fr, 1.1fr, 1.1fr, 1.1fr, 1.3fr, 1.2fr, 1.2fr, 1.2fr),
    stroke: 0.5pt + luma(180),
    align: center,
    [#strong[$B$]], [#strong[Train]], [#strong[Val]], [#strong[Test]], [#strong[Distinct]], [#strong[OOB Acc]], [#strong[Disagr.]], [#strong[Margin]],
    [1], [0.767], [0.488], [0.513], [154.00], [0.349], [0.000], [1.000],
    [5], [0.971], [0.600], [0.625], [153.00], [0.367], [0.781], [0.260],
    [10], [0.996], [0.713], [0.775], [152.70], [0.371], [0.792], [0.251],
    [25], [1.000], [0.775], [0.788], [151.96], [0.646], [0.777], [0.279],
    [50], [1.000], [0.788], [0.838], [150.96], [0.675], [0.776], [0.290],
    [*100*], [*1.000*], [*0.850*], [*0.888*], [*152.36*], [*0.763*], [*0.783*], [*0.284*],
    [200], [1.000], [0.825], [0.875], [152.10], [0.771], [0.774], [0.294],
  ),
  caption: [Bagging ensemble metrics over ensemble sizes $B$. Empirical distinct sample counts match theoretical expectation (151.89).]
) <tab_bagging_sweep>

As demonstrated in @tab_bagging_sweep, the measured distinct sample counts closely match the theoretical expectation across all $B$ (e.g., 152.36 at $B=100$).

== Ensemble Diagnostics & Convergence

To evaluate internal ensemble dynamics, we compute three metrics across $B$:
1. *Out-of-Bag (OOB) Accuracy:* Evaluated on samples $x_i$ using only trees $T_i = {b mid i in.not I_b}$.
2. *Average Pairwise Disagreement:* The fraction of conflicting predictions between tree pairs on validation data:
$ "Disagr." = 1 / (binom(B, 2) M) sum_(b_1 < b_2) sum_(i=1)^M bb(1) [h_(b_1)(x_i) != h_(b_2)(x_i)] $
3. *Mean Vote Margin:* For each sample, the difference between the winning vote fraction and the runner-up:
$ "Margin"(x) = V_((1))(x) - V_((2))(x) $

*Do Diagnostics Tell the Same Story?* As plotted in @fig1_diagnostics, these three metrics tell *complementary, consistent aspects of convergence*:
- *OOB Accuracy* reflects effective generalization capacity; it surges from 34.88% at $B=1$ to 76.25% at $B=100$, tracking validation accuracy closely and confirming that unseen performance saturates once sample coverage stabilizes.
- *Pairwise Disagreement* remains nearly constant around 77.4% to 79.2%. This constancy is vital: bootstrap perturbation maintains high base-classifier diversity throughout training, preventing ensemble collapse.
- *Vote Margin* drops from 1.0 (trivial single tree) to 0.25 at $B=10$ and steadily expands to $0.284 - 0.294$ as $B -> 100$. A widening margin indicates that ensemble consensus becomes increasingly decisive on correct classes.

*Ensemble Selection:* Validation accuracy peaks at *B = 100* (Val Acc: 85.00%, Test Acc: 88.75%). Beyond $B=100$, returns diminish (Val Acc: 82.50% at $B=200$). We freeze $B=100$.

== Robustness & Error Taxonomy

We benchmark the frozen single tree versus the final 100-tree bagged model on clean and occluded test sets:
- *Single Tree:* Clean Test: *58.75%* (Conf: 1.0000) $->$ Masked Test: *30.00%* (Conf: 1.0000). Accuracy drops by 28.75%.
- *Bagged Model (B=100):* Clean Test: *88.75%* (Conf: 0.3914) $->$ Masked Test: *77.50%* (Conf: 0.2923). Accuracy drops by only 11.25%.

*Case Study Analysis (@fig2_cases):*
- *Bagging Corrects Tree (Rows 1--2):* For Face \#1 (Class 0) and Face \#4 (Class 2), the single tree overfits to incidental pixel artifacts, predicting Class 26 and Class 9 with false 100% confidence. Bagging aggregates votes across 100 trees, wherein ground-truth classes emerge as the decisive plurality winners (20 and 19 votes).
- *Both Models Fail (Rows 3--4):* For Face \#7 (Class 3, both predict Class 39 and Class 12) and Face \#13 (Class 6, both predict Class 38 and Class 10), both models fail. These faces possess non-standard poses or dark lighting conditions that are fundamentally unrepresented in the training subset.

*What Bagging Can vs. Cannot Reduce:*
- *Reducible Error:* Bagging eliminates *variance error*. High-dimensional trees are unstable; slight training set perturbations alter the early split choices dramatically. Averaging smooths these idiosyncratic decision boundaries.
- *Irreducible Error:* Bagging *cannot reduce bias error* or systematic domain shifts. When an entire identity's discriminatory features are occluded or absent in training, all trees share the same fundamental blind spots; voting merely averages defective priors.

#figure(
  image("../figures/fig3_q2b_boosting_dynamics.png", width: 95%),
  caption: [SAMME AdaBoost training dynamics over 100 rounds ($d=4$): (a) Weighted training error $epsilon_t$, (b) Learner weight $alpha_t$, (c) Maximum sample weight $max_i w_i^((t))$, and (d) Total weight carried by the 10 heaviest training images.]
) <fig3_boosting_dynamics>

= Question 2: Multiclass Boosting from Scratch

== SAMME Algorithm & Toy Verification

We implement the SAMME (Stagewise Additive Modeling using a Multi-class Exponential loss) algorithm from scratch. For $K=40$ classes, the learner weight update is:
$ alpha_t = log((1 - epsilon_t) / epsilon_t) + log(K - 1) $
where $epsilon_t = sum_(i=1)^N w_i^((t)) bb(1) [y_i != h_t(x_i)]$.
Observation weights are updated via:
$ w_i^((t+1)) = w_i^((t)) exp(alpha_t bb(1) [y_i != h_t(x_i)]) $
and normalized so that $sum_(i=1)^N w_i^((t+1)) = 1$. Prediction probabilities use normalized weighted votes:
$ q_k(x) = S_k(x) / (sum_t alpha_t), quad S_k(x) = sum_t alpha_t bb(1) [h_t(x) = k] $

*Weak Learner Safeguard:* If a weak learner yields weighted error $epsilon_t >= (K-1)/K = 39/40 = 0.975$, its performance is no better than random guessing ($alpha_t <= 0$). Our implementation detects this condition and early-terminates the boosting sequence to preserve margin guarantees.

*Toy 3-Class Verification ($N=6, K=3$):*
To verify correctness before training on faces, we create a 6-sample, 3-class toy dataset. Uniform initial weights are $w_i = 1/6 = 0.1667$. A depth-1 decision stump correctly classifies 4 samples and misclassifies 2 ($epsilon_1 = 2/6 = 0.3333$). Since $epsilon_1 < (3-1)/3 = 0.6667$, the weak learner is valid.
$ alpha_1 = log((1 - 0.3333)/0.3333) + log(3 - 1) = log(2) + log(2) = 1.3863 $
Weights update as:
- Correct samples ($i in {0, 1, 2, 3}$): $w_i -> 0.0833$ (halved).
- Misclassified samples ($i in {4, 5}$): $w_i -> 0.3333$ (doubled).
Sum of normalized weights is exactly 1.0000, validating our mathematical implementation.

== Hyperparameter Tuning on Faces

We evaluate weak learner depths $d in {1, 2, 3, 4}$ and boosting rounds $T in {10, 25, 50, 100}$ on Olivetti Faces.

#figure(
  table(
    columns: (1fr, 1.1fr, 1.1fr, 1.1fr, 1.1fr, 1.2fr),
    stroke: 0.5pt + luma(180),
    align: center,
    [#strong[Depth]], [#strong[Rounds]], [#strong[Train]], [#strong[Val]], [#strong[Test]], [#strong[Time (s)]],
    [1], [10], [0.050], [0.038], [0.050], [0.45],
    [1], [50], [0.096], [0.063], [0.063], [2.25],
    [1], [100], [0.133], [0.125], [0.125], [4.54],
    [2], [10], [0.117], [0.075], [0.088], [0.90],
    [2], [50], [0.271], [0.200], [0.150], [4.55],
    [2], [100], [0.325], [0.175], [0.200], [9.15],
    [3], [10], [0.333], [0.200], [0.238], [1.36],
    [3], [50], [0.417], [0.225], [0.263], [6.75],
    [3], [100], [0.863], [0.363], [0.400], [13.56],
    [4], [10], [0.379], [0.188], [0.250], [1.78],
    [4], [50], [0.971], [0.438], [0.413], [8.93],
    [*4*], [*100*], [*1.000*], [*0.613*], [*0.588*], [*17.92*],
  ),
  caption: [SAMME Boosting grid search. Selected model: depth 4, 100 rounds.]
) <tab_boosting_grid>

*Selected Model:* Depth 4 with 100 rounds achieves the highest validation accuracy (*61.25%*, Test: 58.75%). Stumps ($d=1$) completely fail on 40 classes (12.5% test accuracy after 100 rounds).

*Training Dynamics (@fig3_boosting_dynamics):*
- Weighted training error $epsilon_t$ stabilizes around $0.2 - 0.4$, well below the random threshold 0.975.
- Learner weight $alpha_t$ remains positive throughout, ranging between 4.5 and 6.0.
- Sample weights concentrate aggressively: the maximum individual weight rises to 0.0538 (12.9x uniform weight), and the 10 heaviest training images absorb over 25% of total probability mass.

#figure(
  image("../figures/fig4_q2c_heavy_sample.png", width: 95%),
  caption: [Inspection of the heaviest training sample (Face \#55, Class 9). The sample's weight escalates by 12.9x because weak learners repeatedly fail to classify its atypical facial expression (88% misclassification rate).]
) <fig4_heavy_sample>

== Boosting vs. Bagging Comparison

We compare Boosting ($d=4$) against Bagging ($d="None"$) across 25, 50, and 100 learners:

#figure(
  table(
    columns: (1fr, 1.2fr, 1.1fr, 1.1fr, 1.2fr, 1.1fr, 1.1fr),
    stroke: 0.5pt + luma(180),
    align: center,
    [#strong[$M$]], [#strong[Boost Tr.]], [#strong[Val]], [#strong[Test]], [#strong[Bag Tr.]], [#strong[Val]], [#strong[Test]],
    [25], [0.771], [0.350], [0.363], [1.000], [0.775], [0.788],
    [50], [0.679], [0.275], [0.338], [1.000], [0.788], [0.838],
    [100], [1.000], [0.613], [0.588], [1.000], [0.850], [0.888],
  ),
  caption: [Performance across ensemble sizes $M in {25, 50, 100}$. Bagging consistently dominates Boosting across all test evaluations.]
) <tab_comp_bag_boost>

*Heaviest Sample Analysis (@fig4_heavy_sample):*
Face \#55 (Class 9) achieves the highest final weight in the dataset ($w = 0.0538$, 12.9x uniform). Tracking its round-by-round trajectory reveals that weak learners misclassified this sample in *88% of all boosting rounds*. Because this image features an atypical facial expression and extreme gaze angle compared to other Class 9 training photos, the exponential weight multiplier repeatedly forces subsequent weak learners to attempt separating it, skewing hyperplanes away from canonical facial modes.

#figure(
  image("../figures/fig5_q2d_masked_diff_face.png", width: 95%),
  caption: [Differential response on occluded Face \#0: Bagging correctly identifies ground truth (Class 0) by aggregating robust non-occluded features, whereas Boosting misclassifies it as Class 4 due to high sensitivity to masked pixels.]
) <fig5_diff_face>

== Occlusion Differential Response

On occluded test faces, Bagging and Boosting respond in stark contrast:
- *Face \#0 (Class 0):* As shown in @fig5_diff_face, Bagging correctly predicts Class 0 on the masked face with 33% plurality vote, whereas Boosting misclassifies it as Class 4.
- *Underlying Mechanism:* Bagging trees use independent bootstrap feature subspaces; many trees make splits on peripheral cues (hairline, chin, jawline) that remain intact. Boosting, by contrast, relies on a linear combination of stumps/shallow trees that were heavily tuned to central high-gradient features; when these central pixels become zero, the weighted sum shifts catastrophically.

#figure(
  image("../figures/fig6_q3b_bias_variance.png", width: 95%),
  caption: [Empirical bias-variance decomposition across tree depths (Shallow: $d=4$, Deep: $d=16$, Selected: $d="None"$) comparing Single Trees vs. 100-Tree Bagging. Bagging eliminates 83.0%, 91.7%, and 96.0% of estimation variance, respectively.]
) <fig6_bias_variance>

= Question 3: Bias and Variance as Measured Quantities

== Empirical Resampling Protocol

To rigorously measure bias and variance without synthetic assumptions, we perform $R=30$ refits. Following the required protocol, we sample with replacement, within each identity, 6 faces from that identity's original 6 training faces:
$ S_r = union.big_(c=0)^(39) "Bootstrap"(D_c, N=6), quad r = 1, dots, 30 $
All models are refit on each resampled set without re-tuning. For repetition $r$, $p_r(x_i) in bb(R)^(40)$ represents the predicted class vector on the fixed clean test set ($N=80$):
- *Single Tree:* Probability distribution vector from leaf frequencies.
- *Bagging:* Normalized ensemble vote vector.
- *Boosting:* Normalized weighted vote vector $q_k(x_i)$.

Let $overline(p)_i = 1/R sum_(r=1)^R p_r(x_i)$ and $e_i$ be the one-hot target. We compute:
$ "Bias"^2 = 1 / N sum_(i=1)^N ||overline(p)_i - e_i||_2^2 $
$ "Variance" = 1 / (N R) sum_(i=1)^N sum_(r=1)^R ||p_r(x_i) - overline(p)_i||_2^2 $
This formulation satisfies the exact vector MSE identity:
$ "Total MSE" = "Bias"^2 + "Variance" $

#figure(
  table(
    columns: (1.5fr, 1.1fr, 1.1fr, 1.1fr, 1.3fr),
    stroke: 0.5pt + luma(180),
    align: center,
    [#strong[Model]], [#strong[Bias#super[2]]], [#strong[Variance]], [#strong[Total MSE]], [#strong[Test Acc (Mean $plus.minus$ Std)]],
    [Single Tree (None)], [0.3888], [0.6529], [1.0417], [0.4792 $plus.minus$ 0.0560],
    [*Final Bagging (B=100)*], [*0.5105*], [*0.0253*], [*0.5358*], [*0.7717 $plus.minus$ 0.0412*],
    [Boosting ($d=4, T=100$)], [0.8843], [0.0184], [0.9027], [0.3325 $plus.minus$ 0.0768],
  ),
  caption: [Measured Bias#super[2], Variance, and Test Accuracy across 30 resampled sets.]
) <tab_q3_summary>

== Depth Trajectory of Bias & Variance

We repeat the 30-refit decomposition for single trees and 100-tree bagging across three depths: Shallow ($d=4$), Deep ($d=16$), and Selected ($d="None"$).

#figure(
  table(
    columns: (1.3fr, 1fr, 1fr, 1fr, 1fr, 1.2fr, 1.1fr),
    stroke: 0.5pt + luma(180),
    align: center,
    [#strong[Depth]], [#strong[Tree Bias#super[2]]], [#strong[Tree Var]], [#strong[Bag Bias#super[2]]], [#strong[Bag Var]], [#strong[Var Removed]], [#strong[Acc Gain]],
    [Shallow ($d=4$)], [0.8530], [0.0774], [0.8852], [0.0131], [83.0%], [+19.83%],
    [Deep ($d=16$)], [0.6504], [0.2472], [0.6670], [0.0205], [91.7%], [*+35.79%*],
    [Selected (None)], [0.3857], [0.6518], [0.5073], [0.0258], [*96.0%*], [+28.79%],
  ),
  caption: [Bias-variance decomposition across depths. The largest accuracy gain occurs at Depth 16, not the unpruned tree.]
) <tab_depth_sweep>

As quantified in @tab_depth_sweep and @fig6_bias_variance:
1. *Tree Variance Scales with Depth:* Tree variance explodes from 0.0774 at depth 4 to 0.6518 at depth None.
2. *Bagging Variance Elimination:* Bagging suppresses variance at all depths, removing 83.0% at depth 4, 91.7% at depth 16, and 96.0% at depth None.

== Does Deepest Tree Yield Largest Gain?

*Crucial Finding:* The deepest (unpruned) tree is *NOT* the depth at which bagging yields the largest improvement in test accuracy.
- At $d=16$, Bagging increases test accuracy from 27.87% to 63.67%, an extraordinary gain of *+35.79%*.
- At $d="None"$, Bagging improves test accuracy from 48.12% to 76.92%, a gain of *+28.79%*.

*Mathematical Rationale:*
1. *Leaf Occupancy & Singularity:* At $d="None"$, 11.11% of leaves are singletons. These splits overfit to isolated pixel noise, creating high pairwise disagreement (78.34%). However, because unpruned base trees already classify 48.1% of test samples correctly on their own, the consensus mechanism corrects marginal samples rather than transforming the entire feature partition.
2. *Margin Thresholding:* At $d=16$, base trees possess balanced leaves (14.1 samples/leaf, 0% singletons) that capture authentic facial structural clusters, but individual trees produce unstable margin estimates. Bagging stabilizes these margins, converting widespread near-misses into decisive pluralities, generating the maximal +35.79% accuracy surge.

#figure(
  table(
    columns: (1.5fr, 1fr, 1.1fr, 1fr, 1fr, 1fr),
    stroke: 0.5pt + luma(180),
    align: center,
    [#strong[Model]], [#strong[Clean Acc]], [#strong[Masked Acc]], [#strong[Bias#super[2]]], [#strong[Variance]], [#strong[Total MSE]],
    [Single Tree Baseline], [0.5875], [0.3000], [0.3888], [0.6529], [1.0417],
    [*Final Bagging (B=100)*], [*0.8875*], [*0.7750*], [*0.5105*], [*0.0253*], [*0.5358*],
    [Selected Boosting], [0.5875], [0.5000], [0.8843], [0.0184], [0.9027],
  ),
  caption: [Final Master Synthesis Table across models and perturbation regimes.]
) <tab_master_synthesis>

== Synthesis & Comparative Discussion

We synthesize our complete experimental findings across @tab_master_synthesis:

1. *Which Model is Most Stable?*
   *Bagging ($B=100$)* is by far the most stable model across dataset resamples. Its standard deviation of test accuracy across 30 refits is minimal (0.0412, compared to 0.0768 for boosting and 0.0560 for single trees), and its variance (0.0253) is 25x lower than the baseline tree.

2. *Which Model is Most Sensitive to Occlusion?*
   *Single Decision Tree* is the most sensitive model, collapsing from 58.75% to 30.00% (a 28.75% plunge). A single split along an occluded pixel instantly sends test samples down an erroneous sub-branch. Boosting is also highly vulnerable (58.75% to 50.00%) due to its reliance on sensitive feature stumps. Bagging exhibits remarkable occlusion resilience, retaining *77.50%* accuracy.

3. *Does Lowest Variance Imply Best Accuracy?*
   *No.* Boosting achieves the lowest variance (0.0184), yet its test accuracy is poor (33.25% to 58.75%). Boosting's low variance stems from the rigidity of depth-4 stumps on 40 classes; its bias squared (0.8843) dominates total error. Accuracy requires #strong[minimizing Total MSE ($"Bias"^2 + "Variance"$)], where Bagging achieves the optimal trade-off (0.5358).

= Conclusion & Reproducibility

Through rigorous experimentation on the Olivetti Faces benchmark, we have demonstrated:
1. Bagging unpruned decision trees reduces estimation variance by 96.0%, raising clean face recognition accuracy from 58.75% to 88.75%.
2. SAMME boosting with shallow trees suffers from severe representational bias on multiclass image tasks, concentrating excessive weight on atypical samples (up to 12.9x).
3. The largest accuracy gain from bagging occurs at intermediate depth (depth 16, gain +35.79%), not at unconstrained depth.

*Exact Reproduction Commands:*
```
pip install -r requirements.txt
python3 -m experiments.run_all
```
All code, splits, configs, figures, and data logs are open-sourced in the accompanying repository.

#v(0.8em)
#text(8.5pt)[
  #strong[References]
  #v(0.3em)
  [1] L. Breiman. Bagging predictors. _Machine Learning_, 24(2):123--140, 1996. \
  [2] Y. Freund and R. E. Schapire. A decision-theoretic generalization of on-line learning and an application to boosting. _JCSS_, 55(1):119--139, 1997. \
  [3] J. Zhu, H. Zou, S. Rosset, and T. Hastie. Multi-class AdaBoost. _Statistics and Its Interface_, 2(3):349--360, 2009. \
  [4] P. Domingos. A unified bias-variance decomposition. In _ICML_, pages 231--238, 2000. \
  [5] F. Samaria and S. Young. HMM-based architecture for face identification. _Image and Vision Computing_, 12(8):537--542, 1994.
]
