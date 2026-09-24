# SKAB benchmark — 40 real experiments, 25,931 test rows

37% of test rows are labelled faulty, so an alarm that is *always on* already scores F1 = 0.54 (with FAR = 100%). Always read F1 next to FAR and false-alarm episodes.

| Model | Family | Setting | F1 ↑ | FAR % ↓ | MAR % ↓ | Faults caught | False-alarm episodes / h ↓ | Median delay |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Fixed 3σ limit (today's alarms) | baseline | plain | 0.19 | 3.3 | 88.9 | 34/40 | 23.6 | 70 s |
| Fixed 3σ limit (today's alarms) | baseline | improved (grouped CV) | 0.17 | 3.0 | 90.2 | 25/40 | 9.0 | 106 s |
| PCA T²+Q | statistical | plain | 0.59 | 1.6 | 56.5 | 38/40 | 22.5 | 38 s |
| PCA T²+Q | statistical | improved (grouped CV) | 0.62 | 4.9 | 51.3 | 32/40 | 7.3 | 50 s |
| Isolation Forest | classical ML | plain | 0.00 | 0.3 | 99.9 | 2/40 | 1.5 | 79 s |
| Isolation Forest | classical ML | improved (grouped CV) | 0.30 | 10.3 | 79.0 | 37/40 | 23.6 | 51 s |
| Local Outlier Factor | classical ML | plain | 0.25 | 1.6 | 85.2 | 19/40 | 5.5 | 124 s |
| Local Outlier Factor | classical ML | improved (grouped CV) | 0.41 | 8.2 | 70.6 | 37/40 | 19.4 | 45 s |
| One-Class SVM | classical ML | plain | 0.51 | 9.1 | 60.1 | 39/40 | 23.4 | 47 s |
| One-Class SVM | classical ML | improved (grouped CV) | 0.51 | 9.5 | 60.3 | 36/40 | 11.0 | 60 s |
| Dense autoencoder (MLP) | deep learning | plain | 0.48 | 4.7 | 65.8 | 35/40 | 17.4 | 70 s |
| Dense autoencoder (MLP) | deep learning | improved (grouped CV) | 0.60 | 8.3 | 51.1 | 39/40 | 12.1 | 54 s |
| Conv-1D autoencoder | deep learning | plain | 0.48 | 13.8 | 60.5 | 30/40 | 18.5 | 47 s |
| Conv-1D autoencoder | deep learning | improved (grouped CV) | 0.48 | 20.7 | 57.3 | 31/40 | 7.9 | 48 s |
| LSTM autoencoder | deep learning | plain | 0.29 | 5.7 | 81.5 | 37/40 | 26.2 | 59 s |
| LSTM autoencoder | deep learning | improved (grouped CV) | 0.49 | 12.5 | 60.4 | 37/40 | 16.5 | 55 s |
| Ensemble (5 unsupervised models) | ensemble | plain | 0.59 | 16.7 | 45.9 | 40/40 | 63.9 | 30 s |
| Ensemble (5 unsupervised models) | ensemble | improved (grouped CV) | 0.56 | 12.3 | 53.5 | 39/40 | 11.2 | 68 s |
| Gradient boosting on labelled faults | supervised ML | leave-one-experiment-out | 0.59 | 4.2 | 55.5 | 34/40 | 3.8 | 61 s |
| Gradient boosting on labelled faults | supervised ML | unseen fault family | 0.27 | 5.1 | 82.7 | 22/40 | 5.7 | 130 s |
| Gradient boosting on labelled faults | supervised ML | past experiments only | 0.43 | 4.0 | 70.7 | 24/40 | 2.0 | 78 s |
| ARK Predict v2 (supervised + autoencoder ensemble) | ours (hybrid) | leave-one-experiment-out | 0.61 | 13.7 | 46.4 | 39/40 | 11.5 | 46 s |
| ARK Predict v2 (supervised + autoencoder ensemble) | ours (hybrid) | unseen fault family | 0.57 | 14.0 | 51.1 | 39/40 | 12.1 | 48 s |
| ARK Predict v2 (supervised + autoencoder ensemble) | ours (hybrid) | past experiments only | 0.58 | 14.2 | 49.3 | 39/40 | 11.2 | 46 s |
| ARK Predict live pipeline (v1) | ours | as deployed | 0.47 | 15.3 | 61.6 | 38/40 | 10.4 | 32 s |

Threshold-free ranking quality — pooled ROC-AUC (1.0 = perfect, 0.5 = coin flip):

| Model | raw | drift-robust τ=120 s | drift-robust τ=600 s |
|---|---:|---:|---:|
| Fixed 3σ limit (today's alarms) | 0.590 | 0.605 | 0.594 |
| PCA T²+Q | 0.774 | 0.776 | 0.772 |
| Isolation Forest | 0.662 | 0.665 | 0.662 |
| Local Outlier Factor | 0.706 | 0.712 | 0.713 |
| One-Class SVM | 0.719 | 0.707 | 0.704 |
| Dense autoencoder (MLP) | 0.771 | 0.761 | 0.760 |
| Conv-1D autoencoder | 0.685 | 0.681 | 0.685 |
| LSTM autoencoder | 0.694 | 0.673 | 0.693 |
| ensemble | 0.741 | 0.743 | 0.742 |
| supervised | 0.810 | — | — |
| supervised_lfo | 0.611 | — | — |
| supervised_past | 0.674 | — | — |

Published SKAB leaderboard (same protocol):

| Algorithm | F1 | FAR % | MAR % |
|---|---:|---:|---:|
