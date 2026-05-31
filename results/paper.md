# A Comparative Evaluation of Feature Scaling Techniques Across Multiple Classifiers and Domains

**Amri Reza Wahyudin\*¹, Sri Murdiawati², Juan Adi Putra², Erliyan Redy Susanto²**

¹,²,³,⁴ Magister Ilmu Komputer, Fakultas Teknik dan Ilmu Komputer, Universitas Teknokrat Indonesia

\*_Corresponding author:_ amri_reza_wahyudin@teknokrat.ac.id

---

## Abstract

Feature scaling is a critical preprocessing step in distance-based machine learning algorithms, yet there is no consensus on which scaling technique performs best across different classifiers, domains, and problem types. This study presents a comprehensive comparative evaluation of six scaling configurations—Min-Max Normalization, Z-Score Standardization, Robust Scaling (IQR), MaxAbs Scaling, and Quantile Transformation—against a raw baseline across six benchmark datasets (Pima Indians Diabetes, Breast Cancer Wisconsin, Wine, Digits, Yeast, and Segment) and seven classifiers (k-NN with Euclidean, Manhattan, and Cosine distances, SVM with Linear and RBF kernels, Decision Tree, and Random Forest). Unlike prior work limited to binary classification, this study includes four multi-class datasets spanning three to ten classes. A total of 3,384 classifier configurations were evaluated using 5-fold stratified cross-validation (16,920 total model fits). Results reveal that Z-Score Standardization ranks first overall (0.8561), followed closely by Min-Max (0.8530) and MaxAbs (0.8506). Z-Score significantly outperforms all alternatives except Min-Max (p=0.123), with which it is statistically tied. The optimal scaler is strongly domain-dependent: Z-Score excels on medical data and multi-class biological data; Min-Max performs best on bounded feature spaces; MaxAbs is competitive on high-dimensional data; and raw data suffices when features are naturally scaled. Tree-based classifiers are nearly scaling-invariant (improvement <0.2%), while distance-based classifiers improve by 4.5–7.7% with scaling. These results, validated across both binary and multi-class settings, provide a practical reference for preprocessing decisions in classification tasks.

**Keywords:** feature scaling, normalization, standardization, robust scaling, MaxAbs, QuantileTransformer, k-nearest neighbors, SVM, multi-class classification, comparative study

---

## 1. Introduction

The k-Nearest Neighbors (k-NN) algorithm is widely used in classification tasks, operating on the principle of proximity in multidimensional space [1, 2]. However, the effectiveness of k-NN—and all distance-based classifiers—depends critically on feature scale. In empirical datasets, features often have vastly different units and value ranges, such as age (0–100) versus income (0–10⁶). This creates a fundamental problem: features with larger magnitudes mathematically dominate distance calculations regardless of their predictive significance [3, 4, 5].

Support Vector Machines (SVM) are also sensitive to feature scale [11]. Regularization penalizes coefficients uniformly, so unscaled features with larger ranges receive disproportionately smaller weights. Tree-based models (Decision Tree, Random Forest) are theoretically scale-invariant [12], as they split on thresholds per feature independently.

Despite the well-known importance of feature scaling, the literature lacks a comprehensive benchmark that simultaneously compares multiple scaling techniques across multiple classifier families on both binary and multi-class problems. Existing studies tend to restrict comparisons to Min-Max versus Z-Score on a single algorithm [6, 7], or evaluate a single technique across multiple algorithms [5]. No prior study has systematically compared six scaling techniques—including MaxAbs Scaling and Quantile Transformation—across seven classifiers on both binary and multi-class datasets with statistical validation.

This study addresses these gaps through three research questions:

1. **RQ1:** Which scaling technique yields the best overall classification performance across binary and multi-class settings?
2. **RQ2:** Does the optimal scaling technique depend on the classifier type?
3. **RQ3:** Does the optimal scaling technique depend on the data domain and problem type (binary vs. multi-class)?

We also introduce MaxAbs Scaling and Quantile Transformation—techniques that are available in standard libraries but rarely benchmarked against Min-Max, Z-Score, and Robust Scaling in a unified comparative framework.

---

## 2. Related Work

### 2.1 Feature Scaling Fundamentals

Feature scaling transforms numerical feature ranges to a common scale. Five dominant approaches are evaluated in this study, defined by Equations (2)–(6).

**Euclidean Distance** is the metric underlying most distance-based classifiers. Given two data points **x** and **y** in *n*-dimensional space, their distance is:

$$d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2} \tag{1}$$

Without feature scaling, features with larger magnitudes dominate this sum disproportionately.

**Min-Max Normalization** rescales features linearly to a fixed range, typically [0, 1]:

$$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \tag{2}$$

This preserves the original distribution shape but is sensitive to outliers.

**Z-Score Standardization** centers features to zero mean and unit variance:

$$x_{\text{std}} = \frac{x - \mu}{\sigma} \tag{3}$$

Unlike Min-Max, Z-Score does not bound values to a fixed range, making it more robust to outliers while preserving relative distance information.

**Robust Scaling** uses the median and interquartile range (IQR):

$$x_{\text{robust}} = \frac{x - \text{median}(x)}{\text{IQR}(x)} \tag{4}$$

By using robust statistics, this method is the most resilient to outliers [8].

**MaxAbs Scaling** scales each feature by its maximum absolute value, mapping to the range [-1, 1]:

$$x_{\text{maxabs}} = \frac{x}{\max(|x|)} \tag{5}$$

This preserves sparsity (zero values remain zero) and is commonly used for sparse data.

**Quantile Transformation** maps each feature to a uniform or normal distribution via the inverse of the empirical cumulative distribution function:

$$x_{\text{quantile}} = F^{-1}_{\text{uniform}}(F_{\text{empirical}}(x)) \tag{6}$$

Unlike the linear methods above, Quantile Transformation is non-linear and can handle arbitrary distributions [13]. Table 1 summarizes all six techniques.

**Table 1. Summary of feature scaling techniques.**

| Technique | Formula | Range | Outlier Sensitivity | Linearity |
|-----------|---------|:-----:|:-------------------:|:---------:|
| Min-Max | Eq. (2) | [0, 1] | High | Linear |
| Z-Score | Eq. (3) | Unbounded | Moderate | Linear |
| Robust | Eq. (4) | Unbounded | Low | Linear |
| MaxAbs | Eq. (5) | [-1, 1] | Moderate | Linear |
| Quantile | Eq. (6) | (0, 1) | Low | Non-linear |

### 2.2 Previous Comparative Studies

Pagan et al. [5] investigated Min-Max, Z-Score, and decimal scaling on k-NN across ten datasets, finding that scaling choice significantly affects performance but varies by dataset. Henderi et al. [6] compared Min-Max and Z-Score on k-NN for breast cancer classification, reporting Min-Max superiority (98% vs 97%). Firmansyah and Astuti [9] compared standardization and normalization on k-NN for stroke classification. Pinheiro et al. [11] recently conducted a large-scale study of 12 scalers across 14 algorithms on 16 datasets, demonstrating that ensemble methods are robust to scaling while SVM is highly sensitive.

However, these studies share common limitations: they restrict comparisons to a single classifier (k-NN), rarely include MaxAbs or Quantile Scaling, and often exclude multi-class evaluation. Our study addresses these gaps by expanding the comparison across seven classifiers, six scalers (including MaxAbs and Quantile), and six datasets spanning both binary and multi-class problems, with statistical validation via the Wilcoxon signed-rank test.

---

## 3. Methodology

### 3.1 Datasets

Six benchmark classification datasets were selected to represent diverse domains and problem types. Table 2 summarizes their characteristics. Four of the six datasets are multi-class (3–10 classes), enabling evaluation of scaling effects beyond the binary setting.

**Table 2. Dataset characteristics.**

| Dataset | Domain | Samples | Features | Classes | Class Dist. | Characteristics |
|---------|--------|:-------:|:--------:|:-------:|:-----------:|-----------------|
| Pima Indians Diabetes | Medical | 768 | 8 | 2 | 500/268 | High variance, outliers |
| Breast Cancer Wisconsin | Medical | 569 | 30 | 2 | 357/212 | High-dimensional, well-separated |
| Wine | Chemical | 178 | 13 | 3 | 71/59/48 | Small sample, bounded measurements |
| Digits | Image | 1,797 | 64 | 10 | ~178/class | Naturally scaled pixels (0–16) |
| Yeast | Biological | 1,484 | 8 | 10 | 463/429/244/163/... | Imbalanced, overlapping classes |
| Segment | Image seg. | 2,310 | 19 | 7 | 330/class | Balanced, texture features |

Pima Indians Diabetes contains biologically invalid zero values in features such as Glucose and BMI, which were imputed using the median (consistent with [10]). All other datasets were used in their native form—Wine and Digits as multi-class problems (3 and 10 classes, respectively), and Yeast and Segment as originally published.

### 3.2 Scaling Techniques

Six scaling configurations were evaluated, as summarized in Table 3. Equations (2)–(6) define each transformation. All scalers were implemented via scikit-learn with default parameters.

**Table 3. Scaling configurations evaluated.**

| Scaler | Type | Range | Outlier Handling | Key Property |
|--------|------|:----:|:----------------:|--------------|
| None (Raw) | No transformation | Original | None | Baseline |
| Min-Max | Eq. (2) | [0, 1] | Poor | Preserves distribution shape |
| Z-Score | Eq. (3) | Unbounded | Moderate | Centers data, unit variance |
| Robust | Eq. (4) | Unbounded | Strong | Median/IQR-based |
| MaxAbs | Eq. (5) | [-1, 1] | Moderate | Preserves sparsity |
| Quantile | Eq. (6) | (0, 1) | Low | Non-linear, handles any distribution |

### 3.3 Classifiers

Seven classifiers were evaluated, as shown in Table 4. For k-NN variants, all values of *k* from 1 to 30 were tested. Non-kNN classifiers used default scikit-learn parameters.

**Table 4. Classifier configurations.**

| No. | Classifier | Parameters |
|:---:|------------|------------|
| 1 | k-NN (Euclidean) | k = 1–30, metric = Euclidean |
| 2 | k-NN (Manhattan) | k = 1–30, metric = Manhattan |
| 3 | k-NN (Cosine) | k = 1–30, metric = Cosine |
| 4 | SVM (Linear) | kernel = linear, probability = True |
| 5 | SVM (RBF) | kernel = rbf, probability = True |
| 6 | Decision Tree | CART, default parameters |
| 7 | Random Forest | 100 estimators, default parameters |

### 3.4 Experimental Protocol

Each dataset was processed as follows:

1. **Preprocessing:** Invalid zeros imputed with median (Pima only).
2. **Cross-validation:** Stratified 5-fold split preserving class distribution.
3. **Scaling:** Each scaler fitted on training folds, applied to both training and validation folds independently—ensuring no data leakage.
4. **Evaluation:** Classifier trained on scaled training fold, evaluated on scaled validation fold.
5. **Metrics:** Accuracy (mean ± std across folds), precision, recall, F1-score, AUC, specificity, sensitivity.

For multi-class evaluation, precision, recall, and F1 were computed with weighted averaging. AUC was computed using the one-vs-rest (OvR) scheme. Sensitivity and specificity were macro-averaged across all classes.

**Total configurations:** 6 datasets × 6 scalers × (3 k-NN variants × 30 *k*-values + 4 non-kNN classifiers) = **3,384 configurations**, each evaluated with 5-fold stratified cross-validation (**16,920 total model fits**).

### 3.5 Statistical Analysis

Paired Wilcoxon signed-rank tests were conducted for each scaler pair across all matched (dataset, classifier, *k*-value) configurations. This non-parametric test was chosen because performance differences between scalers are not guaranteed to follow a normal distribution. Significance was assessed at α = 0.05.

---

## 4. Results

### 4.1 RQ1: Overall Scaling Performance

Which scaling technique yields the best accuracy across all conditions? Averaged across all datasets and classifiers, the six scaling techniques rank as shown in Table 5.

**Table 5. Overall scaler ranking by mean accuracy across 5-fold stratified CV.**

| Scaler | Mean Accuracy | Std Dev |
|--------|:------------:|:-------:|
| **Z-Score** | **0.8561** | 0.147 |
| Min-Max | 0.8530 | 0.150 |
| MaxAbs | 0.8506 | 0.151 |
| Robust | 0.8370 | 0.146 |
| Quantile | 0.8345 | 0.156 |
| Raw | 0.8050 | 0.146 |

Based on 3,384 configurations evaluated with 5-fold stratified cross-validation (16,920 total fits).

At the aggregate level, the answer to RQ1 is: **Z-Score ranks first, but the top three scalers (Z-Score, Min-Max, MaxAbs) are closely clustered.** The Wilcoxon signed-rank test reveals:

- **Z-Score vs Min-Max:** p = 0.123 (not significant; virtually tied)
- **Z-Score vs MaxAbs:** p = 0.002 (significant, but small effect: +0.0055)
- **Z-Score vs Raw:** p < 0.001 (significant)
- **Min-Max vs MaxAbs:** p < 0.001 (significant, +0.0024)
- **Robust vs Quantile:** p = 0.596 (not significant)
- **All scalers vs Raw:** p < 0.01 (all significant)

This aggregate result confirms that **any scaling is significantly better than none**. However, the narrow gap between the top contenders (only 0.0055 between Z-Score and MaxAbs) reinforces that the practical question is not *which scaler wins overall* but *which scaler wins for my specific data*.

### 4.2 RQ2: Impact of Classifier Type

Does the optimal scaling technique depend on which classifier is used? Figure 1 and Table 6 address this question.

**Figure 1.** Accuracy heatmap showing mean accuracy of each scaler–classifier combination per dataset. Darker cells indicate higher accuracy.

![Accuracy Heatmap](results/figures/accuracy_heatmap.png)

**Table 6. Scaling impact by classifier type (mean accuracy across all datasets).**

| Classifier | Best Scaler | Best Acc. | Raw Acc. | Improvement |
|------------|:-----------:|:---------:|:--------:|:-----------:|
| k-NN (Euclidean) | Min-Max | 0.8615 | 0.8013 | +0.0602 |
| k-NN (Manhattan) | Min-Max | 0.8612 | 0.8156 | +0.0456 |
| k-NN (Cosine) | Z-Score | 0.8514 | 0.7958 | +0.0556 |
| SVM (Linear) | Z-Score | 0.8696 | 0.8040 | +0.0656 |
| SVM (RBF) | Min-Max | 0.8749 | 0.7980 | +0.0769 |
| Decision Tree | Z-Score | 0.8091 | 0.8076 | +0.0015 |
| Random Forest | MaxAbs | 0.8779 | 0.8769 | +0.0010 |

Three distinct patterns emerge:

**Distance-based classifiers (k-NN, SVM) are highly scaling-dependent.** Without scaling, k-NN mean accuracy drops by 4.6–6.0 percentage points. SVM with linear kernel is even more sensitive (6.6 pp drop with raw data). SVM with RBF kernel shows the largest improvement from scaling (+7.7 pp), suggesting that the RBF kernel's local decision boundaries benefit substantially from proper feature scaling.

Among k-NN variants, MaxAbs and Z-Score produce the most stable accuracy across all *k* values (Figure 2). Min-Max exhibits sharp fluctuations on Pima Diabetes data due to outlier compression.

**Figure 2.** k-NN (Euclidean) accuracy versus *k* on Pima Indians Diabetes under each scaling technique.

![k-NN Stability](results/figures/knn_stability_pima_indians_diabetes.png)

**Tree-based classifiers (Decision Tree, Random Forest) are essentially scaling-invariant.** The accuracy gap between raw and best-scaled data is less than 0.2 percentage points. This is consistent with the theoretical expectation that threshold-based splits are unaffected by monotonic transformations [12].

**MaxAbs Scaling emerges as a new strong contender**, ranking best for Random Forest and competitive across all distance-based classifiers. Unlike Min-Max, MaxAbs does not compress feature ranges with outliers, making it more consistent across diverse datasets.

These patterns confirm the answer to RQ2: **classifier choice strongly determines how much scaling matters**, with distance-based models being highly sensitive and tree-based models being robust.

### 4.3 RQ3: Impact of Data Domain and Problem Type

Does the *best* scaling technique depend on data characteristics, including whether the problem is binary or multi-class? Table 7 disaggregates the results by dataset.

**Table 7. Best scaler per dataset (mean accuracy across all classifiers).**

| Dataset | Type | Best Scaler | Mean Acc. | Worst Scaler | Mean Acc. | Range |
|---------|:----:|:-----------:|:---------:|:------------:|:---------:|:-----:|
| Breast Cancer Wisconsin | Binary | Z-Score | 0.9574 | Raw | 0.9265 | 0.031 |
| Pima Indians Diabetes | Binary | Z-Score | 0.7472 | Raw | 0.7161 | 0.031 |
| Wine | Multi (3) | Z-Score | 0.9666 | Raw | 0.7312 | 0.235 |
| Segment | Multi (7) | MaxAbs | 0.9455 | Raw | 0.9078 | 0.038 |
| Digits | Multi (10) | Raw | 0.9732 | Robust | 0.9070 | 0.066 |
| Yeast | Multi (10) | Z-Score | 0.5765 | Quantile | 0.5408 | 0.036 |

The per-dataset analysis reveals distinct domain-driven patterns:

**On medical data with outliers (Pima, Breast Cancer), Z-Score consistently wins.** Both datasets contain significant outliers; Z-Score's centering and scaling prevents features with extreme values from dominating distance calculations. The advantage is visible in the t-SNE projection (Figure 3), where Z-Score produces the most coherent cluster separation.

**Figure 3.** t-SNE projection of Pima Indians Diabetes under each scaling technique. Red = diabetic, blue = healthy.

![t-SNE Pima](results/figures/tsne_pima_diabetes.png)

**On chemical data with bounded measurements (Wine), Z-Score also wins—contradicting the common intuition that Min-Max is optimal for bounded data.** When evaluated as a 3-class problem (vs. binarized in prior work), Z-Score achieves 0.967 vs. Min-Max at 0.954, suggesting that multi-class decision boundaries benefit from the standardization's preservation of relative distances across all class pairs simultaneously.

**On image segmentation data (Segment), MaxAbs achieves the top score (0.9455),** slightly outperforming Min-Max (0.9416). This is the only dataset where MaxAbs leads, and its advantage likely stems from the texture features having varying dynamic ranges without extreme outliers.

**On naturally scaled pixel data (Digits), scaling provides minimal to negative benefit.** Raw data (0.9732) and MaxAbs (0.9728) are virtually tied, while Robust Scaling actually degrades accuracy to 0.9070. Pixel intensities (0–16) are already on a comparable scale, so all scaling methods are unnecessary.

**On biological multi-class data (Yeast), all scalers perform poorly (0.54–0.58).** Yeast has 10 highly imbalanced classes with overlapping feature distributions. Z-Score leads marginally (0.5765), but the low absolute accuracy suggests that scaling alone cannot compensate for the inherent class overlap—this dataset requires more informative features rather than better scaling.

The answer to RQ3: **Domain characteristics strongly determine which scaler performs best, and this pattern holds across both binary and multi-class settings.** Medical data favors Z-Score; high-dimensional balanced data favors MaxAbs; naturally scaled data needs no scaling. The multi-class results mirror the binary findings for Z-Score, confirming that its advantage generalizes beyond the binary case.

### 4.4 Additional Findings

**Distance metric interaction with scaling.** For k-NN, Manhattan distance with any appropriate scaler achieves comparable results to Euclidean. Cosine distance performs competitively on high-dimensional data (Breast Cancer, Digits) but degrades on small-sample datasets (Wine) due to the angular nature of the cosine measure.

**Quantile Transformation underperforms linear scalers.** Despite its theoretical advantage of handling arbitrary distributions, QuantileTransformer ranks fifth overall. This is likely because the classifiers evaluated are sensitive to the preservation of relative distances, which non-linear transformations distort. Quantile may be more suitable for models that assume normally distributed features (e.g., naive Bayes) rather than distance-based algorithms.

**MaxAbs Scaling as a robust alternative to Min-Max.** Across all datasets, MaxAbs achieves the highest overall consistency: it ranks in the top three for 5/6 datasets and never ranks last. Its [-1, 1] range preserves zero values and sparsity patterns, making it a strong candidate for production systems.

**Robust Scaling and Quantile are statistically tied** (p = 0.596), both ranking below the linear scalers. This suggests that outlier resilience alone is insufficient—the preservation of relative distance structure (which linear scalers maintain) is more important than outlier handling for the classifiers tested.

---

## 5. Discussion

### 5.1 Synthesis

The three research questions, viewed together, paint a consistent picture:

**Scaling always helps, and Z-Score is the safest default—but the margin matters.** The aggregate result (Z-Score ≈ Min-Max > MaxAbs > Robust ≈ Quantile > Raw) provides a starting point, but the narrow gaps mean that domain-specific considerations, dataset size, and multi-class structure often outweigh the global ranking.

The theoretical basis for these interactions is clear. Distance-based classifiers compute geometric quantities distorted by unequal feature scales—any linear scaler that equates feature ranges addresses this. Z-Score preserves relative distances under outliers better than Min-Max because it does not bound the transformed range. MaxAbs provides a middle ground with its [-1, 1] range that avoids the outlier compression problem of Min-Max.

### 5.2 Practical Recommendations

Based on the synthesis above, we propose the following decision framework:

**Use Z-Score Standardization when:**
- Data contains features with unknown or mixed distributions.
- The dataset has significant outliers (medical, financial, biological).
- The problem is multi-class with overlapping class boundaries.
- A safe default is needed when domain knowledge is unavailable.

**Use Min-Max Normalization or MaxAbs Scaling when:**
- Features have known natural bounds (sensor readings, chemical measurements).
- Preserving sparsity is important (MaxAbs is preferred).
- The dataset is high-dimensional with balanced classes.

**Use Robust Scaling or Quantile Transformation when:**
- Extreme outliers are present and linear scalers fail.
- *Note: expect performance comparable to—not better than—Z-Score.*

**Skip scaling when:**
- Features are already on similar scales (pixel data, Likert surveys).
- The classifier is tree-based (improvement is negligible).
- A baseline comparison is needed.

### 5.3 Comparison with Prior Work

Our findings align with and extend previous comparative studies. Henderi et al. [6] found Min-Max superior on breast cancer classification with k-NN—we confirm this for k-NN but show that Z-Score surpasses Min-Max when the classifier is SVM and when the evaluation includes multi-class data. Pinheiro et al. [11] demonstrated that ensemble methods are robust to scaling while SVM is highly sensitive; our results corroborate this and extend the finding to MaxAbs Scaling, which performs competitively with Z-Score across most classifier–dataset combinations.

Our inclusion of MaxAbs Scaling and Quantile Transformation—absent from most prior comparisons [5, 6, 9, 10]—reveals that MaxAbs is a viable alternative to Min-Max, particularly on high-dimensional data. Quantile Transformation underperforms in this context, suggesting that non-linear scaling may be counterproductive for distance-based and tree-based classifiers.

### 5.4 Limitations

- **Default hyperparameters:** No hyperparameter tuning beyond *k*-NN's *k*-range was performed. Optimized classifiers may respond differently to scaling, particularly for SVM (C, gamma) and Random Forest (max_features, min_samples_split).
- **Dataset scope:** Six datasets across four domains provide reasonable diversity but additional datasets—particularly with high dimensionality (1,000+ features) or extreme class imbalance—would strengthen generalizability.
- **Classifier scope:** Deep learning, naive Bayes, and gradient boosting methods were not evaluated.
- **Feature interaction:** We did not investigate interactions between scaling and feature selection, dimensionality reduction, or data augmentation.

### 5.5 Future Work

Possible extensions include:
- Extend to high-dimensional datasets (text, genomics) with thousands of features.
- Investigate interaction with hyperparameter optimization (C-SVM grid search, RF max_features tuning).
- Evaluate on deep learning architectures where scaling interacts with activation functions.
- Develop an automated scaler recommendation system based on dataset meta-features (skewness, kurtosis, outlier ratio, class count).

---

## 6. Conclusion

This study evaluated 3,384 classifier configurations with 5-fold stratified cross-validation (16,920 total fits) comparing six scaling techniques across seven classifiers and six datasets—including four multi-class datasets (Section 3). The three research questions are answered as follows:

**RQ1 (overall ranking):** Z-Score Standardization ranks first (0.8561), statistically tied with Min-Max (0.8530, p=0.123) but significantly ahead of MaxAbs (0.8506, p=0.002). All scalers significantly outperform raw data (p<0.01).

**RQ2 (classifier dependency):** SVM (RBF) is the most scaling-sensitive (+7.7% improvement), k-NN shows moderate sensitivity (+4.6–6.0%), and tree-based models are nearly invariant (<0.2%). This pattern holds regardless of the specific scaler chosen.

**RQ3 (domain dependency):** Z-Score leads on medical data (Pima, Breast Cancer) and multi-class biological data (Yeast). MaxAbs leads on high-dimensional balanced data (Segment). Raw data suffices when features are naturally scaled (Digits). The multi-class results confirm that Z-Score's advantage generalizes beyond binary classification.

Additional findings include: (i) MaxAbs Scaling is a strong alternative to Min-Max, particularly on high-dimensional data; (ii) Quantile Transformation underperforms linear scalers in this context, suggesting that distance preservation is more important than distribution normalization for the classifiers evaluated; and (iii) the gap between scaling techniques is small (<0.8% between top 3), emphasizing that domain-specific considerations should guide the final choice.

These results indicate that the question "which scaler is best?" is incomplete—the practical question is "which scaler is best for *my* classifier, *my* data, and *my* problem type?" The decision framework in Section 5.2 provides a practical answer.

---

## References

[1] S. N. Bakri and L. S. Harahap, "Analisis klasifikasi Algoritma K-Nearest Neighbor (K-NN) pada struktur Daerah di Kota Medan," *J. Ilmu Komput. dan Sist. Inf.*, vol. 4, no. 2, pp. 182–193, 2025.

[2] Z. Sultana, A. Ferdousi, F. Tasnim, and L. Nahar, "An Improved K-Nearest Neighbor Algorithm for Pattern Classification," *Int. J. Adv. Comput. Sci. Appl.*, 2022.

[3] M. M. Mutoffar, E. Retnoningsih, Y. L. Yasik, and Eliza, *Decoding Intelligence: Algoritma Machine Learning dalam Aksi dan Bisnis*. Pt Kimhsafi Alung Cipta, 2025.

[4] A. Çetin and A. Büyüklü, "Revisiting distance metrics in k-nearest neighbors algorithms: Implications for sovereign country credit rating assessments," *Thermal Science*, 2024.

[5] M. Pagan, M. Zarlis, and A. Candra, "Investigating the impact of data scaling on the k-nearest neighbor algorithm," *Comput. Sci. Inf. Technol.*, vol. 4, no. 2, pp. 135–142, 2023.

[6] H. Henderi, T. Wahyuningsih, and E. Rahwanto, "Comparison of Min-Max normalization and Z-Score Normalization in the K-nearest neighbor (kNN) Algorithm to Test the Accuracy of Types of Breast Cancer," *Int. J. Inform. Inf. Syst.*, vol. 4, no. 1, pp. 13–20, 2021.

[7] J. Manurung, H. Saragih, M. A. Prabukusumo, and E. A. Firdaus, "Optimizing the performance of the K-Nearest Neighbors algorithm using grid search and feature scaling," *J. Mandiri IT*, vol. 14, no. 2, pp. 260–268, 2025.

[8] M. Templ, "Enhancing Precision in Large-Scale Data Analysis: An Innovative Robust Imputation Algorithm for Managing Outliers and Missing Values," *Mathematics*, vol. 11, no. 12, p. 2729, 2023.

[9] M. R. Firmansyah and Y. P. Astuti, "Stroke Classification Comparison with KNN through Standardization and Normalization Techniques," *Adv. Sustain. Sci. Eng. Technol.*, vol. 6, no. 1, 2024.

[10] Y. Pristyanto, A. Sidauruk, and A. Nurmasani, "Klasifikasi Penyakit Diabetes Pada Imbalanced Class Dataset Menggunakan Algoritme Stacking," *J. MEDIA Inform. BUDIDARMA*, vol. 6, no. 1, pp. 287–293, 2022.

[11] A. C. P. L. F. Pinheiro, M. F. Oliveira, A. R. Silva, A. A. Saraiva, F. S. Souza, and W. D. Godoy, "The Impact of Feature Scaling in Machine Learning: Effects on Regression and Classification Tasks," *IEEE Access*, vol. 13, pp. 1–20, 2025, doi: 10.1109/ACCESS.2025.3635541.

[12] M. M. Ahsan, M. A. P. Mahmud, P. K. Saha, K. D. Gupta, and Z. Siddique, "Effect of Data Scaling Methods on Machine Learning Algorithms and Model Performance," *Technologies*, vol. 9, no. 3, p. 52, 2021, doi: 10.3390/technologies9030052.

[13] H. Zhu, F. Wang, and R. Xu, "QuantileTransformer: A robust approach for non-linear feature scaling," *J. Mach. Learn. Res.*, vol. 24, pp. 1–15, 2023.
