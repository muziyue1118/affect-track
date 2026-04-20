# Online Valence/Arousal Deployment Training

- Network: TSception
- Normalization: per_window_per_channel_zscore
- Filter context: 6.0s, trim 1.0s each side, model window 4.0s
- Device: cuda
- Final artifacts keep the best epoch by test balanced accuracy when a holdout split is enabled.
- If holdout is enabled, test metrics are used for model selection and are therefore deployment-tuning metrics.
- Subject filter: ['sub3']
- Holdout trials: [{'trial_id': 'sub3_negative_1.mp4_45', 'video_name': 'negative_1.mp4', 'category': 'negative', 'valence': 2, 'arousal': 2}, {'trial_id': 'sub3_positive_2.mp4_33', 'video_name': 'positive_2.mp4', 'category': 'positive', 'valence': 5, 'arousal': 5}]

## Tasks

### valence

- Artifact: valence_tsception.pt
- Windows: train 469, test 132
- Trials: train 8, test 2
- Subjects: 1
- Class counts: {'train': {'negative': 219, 'positive': 250}, 'test': {'negative': 82, 'positive': 50}}
- Train accuracy: 1.0000
- Train balanced accuracy: 1.0000
- Train macro F1: 1.0000
- Train confusion matrix [[TN, FP], [FN, TP]]: [[219, 0], [0, 250]]
- Test accuracy: 0.9924
- Test balanced accuracy: 0.9939
- Test macro F1: 0.9920
- Test confusion matrix [[TN, FP], [FN, TP]]: [[81, 1], [0, 50]]
- Best epoch: 74
- Best test loss: 0.030911434441804886
- Best test balanced accuracy: 0.9939024390243902
- Final loss: 0.013977047490576903

### arousal

- Artifact: arousal_tsception.pt
- Windows: train 682, test 132
- Trials: train 11, test 2
- Subjects: 1
- Class counts: {'train': {'low': 272, 'high': 410}, 'test': {'low': 82, 'high': 50}}
- Train accuracy: 0.9927
- Train balanced accuracy: 0.9939
- Train macro F1: 0.9924
- Train confusion matrix [[TN, FP], [FN, TP]]: [[272, 0], [5, 405]]
- Test accuracy: 0.7121
- Test balanced accuracy: 0.7683
- Test macro F1: 0.7115
- Test confusion matrix [[TN, FP], [FN, TP]]: [[44, 38], [0, 50]]
- Best epoch: 80
- Best test loss: 0.8445690274238586
- Best test balanced accuracy: 0.7682926829268293
- Final loss: 0.02036851473067972
