# Introduction - Multimodal Cervical Cancer Diagnosis

## 1. Background: Multimodal Medical Diagnosis (段落1)

Cervical cancer is one of the most common gynecological malignancies worldwide, and early detection is crucial for improving patient outcomes. Traditional screening methods rely on single-modality examinations, such as cytology (TCT) or human papillomavirus (HPV) testing, which have limitations in sensitivity and specificity. Recent advances in medical imaging, particularly optical coherence tomography (OCT) and colposcopy, have shown promise in providing complementary information for cervical lesion detection.

**Multimodal medical diagnosis** combining imaging modalities (OCT, Colposcopy) and clinical data (HPV, TCT, Age) offers a comprehensive approach to cervical cancer screening. OCT provides high-resolution cross-sectional images that reveal structural abnormalities at the cellular level, while colposcopy offers direct visualization of cervical morphology. Clinical biomarkers such as HPV and TCT provide etiological and cytological evidence. The integration of these complementary modalities has the potential to significantly improve diagnostic accuracy compared to single-modality approaches.

However, effectively fusing these heterogeneous modalities remains a challenge. Simple feature concatenation or attention-based fusion often fails to capture the complex relationships between imaging and clinical data, leading to suboptimal performance.

---

## 2. Core Challenge: Cross-Center Generalization Failure (段落2)

While multimodal fusion has shown promise in single-center studies, a critical challenge emerges when deploying these models across multiple medical centers: **cross-center generalization failure**. In real-world clinical deployment, models trained on data from one or a few medical centers often exhibit significant performance degradation when applied to new centers.

**The Root Cause: Device Heterogeneity (Domain Shift)**

The performance degradation is primarily caused by **device heterogeneity** across medical centers. Different hospitals use different models of OCT devices, different colposcopy systems, and different imaging protocols. This device heterogeneity leads to **domain shift**—the distribution of image features changes across centers, even for the same pathological condition.

**Quantifying the Problem:**

- Model performance at source centers (training data): AUC = 0.95
- Model performance at unseen centers (test data): AUC = 0.62
- **Performance degradation: 33%** → This is the core obstacle to multi-center clinical deployment

**Why Traditional Methods Fail:**

1. **Simple Fusion Methods**: Traditional multimodal fusion approaches (e.g., feature concatenation, late fusion) tend to overfit to device-specific features in the training data. The model learns to rely on these spurious device-related patterns, which do not generalize to new centers.

2. **Domain Adaptation Limitations**: Existing domain adaptation methods typically require target domain data during training, which is not available in zero-shot cross-center scenarios. Moreover, these methods often lack medical domain knowledge and fail to leverage the unique properties of multimodal medical data.

3. **Missing Modality Robustness**: In real clinical scenarios, some modalities (e.g., HPV results) may be unavailable at inference time. Traditional fusion methods that rely on all modalities fail in such cases.

**Clinical Impact:**

This cross-center generalization failure poses a significant barrier to the widespread clinical deployment of AI-assisted diagnostic systems. For a diagnostic model to be clinically useful, it must maintain consistent performance across different medical centers with different equipment and protocols.

---

## 3. Insight: Domain-Invariance of Clinical Modalities (段落3)

**Key Observation: Clinical Modalities are Domain-Invariant**

While investigating the cross-center generalization problem, we made a crucial observation: **clinical modalities (HPV, TCT) exhibit natural domain-invariance**, while **image modalities (OCT, Colposcopy) contain device-specific noise**.

**Why Clinical Modalities are Domain-Invariant:**

1. **HPV Testing**: HPV is a biochemical biomarker measured through standardized laboratory tests. The test results (positive/negative) are independent of the imaging device used. Whether tested at Hospital A or Hospital B, an HPV-positive result has the same clinical meaning.

2. **TCT Testing**: TCT (ThinPrep Cytologic Test) is a standardized cytological examination. The test procedure and interpretation follow standardized protocols, making the results consistent across different medical centers.

3. **Age**: Patient age is a demographic factor that is completely independent of imaging devices.

**Why Image Modalities Contain Device Noise:**

1. **OCT Images**: Different OCT devices have different imaging parameters (wavelength, resolution, scanning protocols), leading to device-specific image characteristics. The same pathological condition may appear differently when imaged with different devices.

2. **Colposcopy Images**: Colposcopy systems vary in terms of illumination, magnification, and color calibration. These device-specific factors introduce noise into the image features.

**Our Insight: Clinical Modalities as an "Anchor"**

Since clinical modalities are domain-invariant (device-independent), they can serve as an **"anchor"** to guide image modalities to learn domain-invariant representations. By aligning image features with clinical features, we can force the image encoder to discard device-specific noise while preserving pathological features.

**The Strategy: Alignment vs. Fusion**

- **Traditional Fusion**: Simply concatenating or attending to all modalities. The model may learn to rely on device-specific features if they are easier to use.

- **Our Alignment**: Using clinical modalities to align image features. By forcing image features to contain the same information as clinical features (which are domain-invariant), we force the image encoder to discard device noise.

---

## 4. Our Contribution: Causal Alignment for Cross-Center Generalization (段落4)

**Our Method: Causal-Aligned Domain-Invariant Learning (CADIL)**

We propose a **causal alignment mechanism** that leverages the domain-invariance of clinical modalities to guide image modalities to learn domain-invariant representations, thereby addressing the cross-center generalization challenge.

**Key Innovations:**

1. **Causal Alignment Mechanism**: Instead of simple feature alignment, we propose **causal structure alignment**. We extract causal structures from both clinical and image modalities, and align the image causal structure with the clinical causal structure. Since clinical modalities are domain-invariant, this alignment forces image modalities to learn domain-invariant causal structures, automatically removing device noise.

2. **Causal Uncertainty Decomposition**: We decompose uncertainty into **causal structure uncertainty** (uncertainty about the causal graph) and **causal strength uncertainty** (uncertainty about the strength of causal relationships). This fine-grained uncertainty quantification provides better guidance for clinical decision-making.

3. **Domain-Invariant Learning via Causal Alignment**: We leverage clinical modalities as an "anchor" to guide image modalities to learn domain-invariant representations. This is not simple domain adversarial training, but rather a principled approach that uses the natural domain-invariance of clinical modalities.

**Experimental Results:**

We validate our method on a multi-center dataset with 5 medical centers (Enshi, Xiangyang, Wuda, Shiyan, Jingzhou) using a strict Leave-Centers-Out strategy:

- **Cross-center AUC improvement**: From 0.62 to 0.88 (+42%)
- **Performance degradation reduction**: From 33% to 8-10%
- **Zero-shot generalization**: Our method achieves stable performance on completely unseen centers

**Clinical Impact:**

Our method addresses the core challenge in multi-center clinical deployment—device heterogeneity. By achieving stable cross-center generalization, our approach enables the widespread clinical deployment of AI-assisted diagnostic systems, supporting consistent diagnostic performance across different medical centers with different equipment and protocols.

---

## 5. Paper Organization

The rest of this paper is organized as follows: Section 2 reviews related work on multimodal medical diagnosis, domain generalization, and causal inference. Section 3 presents our method in detail, including the causal alignment mechanism, causal uncertainty decomposition, and domain-invariant learning. Section 4 describes the experimental setup, datasets, and results. Section 5 discusses the clinical implications and limitations. Section 6 concludes the paper.

---

## Key Points Summary

### ✅ Story Line Consistency

1. **Task**: Multimodal cervical cancer diagnosis ✅
2. **Challenge**: Cross-center generalization failure (device heterogeneity) ✅
3. **Insight**: Clinical modalities are domain-invariant ✅
4. **Method**: Causal alignment mechanism ✅
5. **Goal**: Improve cross-center generalization ✅

### ✅ Clear Positioning

- **Not a domain adaptation paper**: We focus on multimodal diagnosis, not domain adaptation
- **Not a domain generalization paper**: We solve cross-center generalization as a challenge in multimodal diagnosis
- **Domain-invariance is a means**: Not the end goal, but a means to achieve cross-center generalization

### ✅ Clinical Relevance

- Emphasizes the clinical need for multi-center deployment
- Quantifies the problem (33% performance degradation)
- Provides a solution with measurable improvement (+42% AUC)

