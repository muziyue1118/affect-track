# Online Valence/Arousal Deployment Training

- Network: FBSTCNet
- Normalization: per_window_per_channel_zscore
- Device: cuda
- Final artifacts are trained on all available non-neutral binary windows.
- This deployment report is not a strict held-out generalization estimate.

## Tasks

### valence

- Artifact: valence_fbstcnet.pt
- Windows: 2112
- Trials: 41
- Subjects: 5
- Class counts: {'negative': 1037, 'positive': 1075}
- Final loss: 0.04913505692754618

### arousal

- Artifact: arousal_fbstcnet.pt
- Windows: 2324
- Trials: 46
- Subjects: 5
- Class counts: {'low': 992, 'high': 1332}
- Final loss: 0.04630647033248862
