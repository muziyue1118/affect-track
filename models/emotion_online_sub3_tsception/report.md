# Online Valence/Arousal Deployment Training

- Network: TSception
- Normalization: per_window_per_channel_zscore
- Device: cuda
- Final artifacts are trained on all available non-neutral binary windows.
- This deployment report is not a strict held-out generalization estimate.
- Subject filter: ['sub3']

## Tasks

### valence

- Artifact: valence_tsception.pt
- Windows: 592
- Trials: 10
- Subjects: 1
- Class counts: {'negative': 301, 'positive': 291}
- Train accuracy: 1.0000
- Train balanced accuracy: 1.0000
- Train macro F1: 1.0000
- Train confusion matrix [[TN, FP], [FN, TP]]: [[301, 0], [0, 291]]
- Final loss: 0.008291925426180425

### arousal

- Artifact: arousal_tsception.pt
- Windows: 803
- Trials: 13
- Subjects: 1
- Class counts: {'low': 352, 'high': 451}
- Train accuracy: 1.0000
- Train balanced accuracy: 1.0000
- Train macro F1: 1.0000
- Train confusion matrix [[TN, FP], [FN, TP]]: [[352, 0], [0, 451]]
- Final loss: 0.036256768131771915
