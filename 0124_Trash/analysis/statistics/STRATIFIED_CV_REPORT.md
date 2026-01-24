# Stratified Cross-Validation Report

- Data path: 5centers_multi
- Number of folds: 5
- Features:
  - Categorical: HPV清洗, TCT清洗
  - Numeric: AGE

## Aggregate Metrics (mean [95% CI])
- **auc**: 0.645 [0.604, 0.680]
- **accuracy**: 0.570 [0.540, 0.594]
- **f1**: 0.540 [0.509, 0.571]
- **sensitivity**: 0.773 [0.718, 0.834]
- **specificity**: 0.471 [0.444, 0.500]
- **ppv**: 0.415 [0.391, 0.434]
- **npv**: 0.813 [0.770, 0.858]
- **biopsy_rate**: 0.609 [0.581, 0.634]
- **recall**: 0.773 [0.716, 0.838]
- **precision**: 0.415 [0.387, 0.435]
- **fold**: 3.000 [1.800, 4.200]

## Fold-level Metrics
- Fold 1: AUC=0.567, Sens=0.687, Spec=0.429, PPV=0.367, NPV=0.740, BiopsyRate=0.609
- Fold 2: AUC=0.684, Sens=0.797, Spec=0.481, PPV=0.425, NPV=0.831, BiopsyRate=0.609
- Fold 3: AUC=0.690, Sens=0.891, Spec=0.466, PPV=0.445, NPV=0.899, BiopsyRate=0.650
- Fold 4: AUC=0.650, Sens=0.769, Spec=0.455, PPV=0.410, NPV=0.800, BiopsyRate=0.619
- Fold 5: AUC=0.632, Sens=0.723, Spec=0.523, PPV=0.427, NPV=0.793, BiopsyRate=0.558