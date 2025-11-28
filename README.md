## Data (/data)

The files contain Multi-Index Headers (hierarchical headers).

1. `tracks.csv`
- **Row 1**: groups the columns, we can ignore everything under "album" and "artist", and focus on "track" (song).
- **Row 2**: the actual column names. 
    - *listens* -> can be used as target for prediction. (either classification or regression)
    - *track_id* -> unique identifier for each track (song).

2. `features.csv`
- **Row 1**: audio algorithm used to extract features
- **Row 2**: audio chnages over time (duration of the song), so the columns represent the summary statistics of the features over time.
    - *mean* -> average value
    - *std* -> standard deviation
    - *skew* -> average value of the first derivative of the feature over time
    - *kurtosis* -> variance of the first derivative of the feature over time
    - *min/max/median* 
- **Row 3**: the algorithms generate multiple features (vector) not just one.
    - e.g., `spectral_contrast` generates 7 values (one per frequency band)
    - e.g., `mfcc` generates 20 values (one per cepstral coefficient)

So, each numbered column in the statistic corresponding to that feature for the whole song.

### Table of the features:
| Feature Name            | What it means |
|-------------------------|-------------------------------|
Chroma STFT               | 12-bin representation of the spectral energy for each pitch class (C, C#, D, ..., B) (Do, re, mi,...)
Tonnetz                  | harmony, detects changes and harmonic
Spectral Centroid       | Brightness, sound "bright" (like a cymbal) or "dark" (like a bass drum)?
Spectral Bandwidth      | Range, how wide is the range of frequencies present in the sound?
Spectral Contrast       | Difference between quiet and loud parts of the spectrum
ZCR (Zero Crossing Rate) | Noisiness, how often the signal changes from positive to negative

## Possible aproaches

1. Use the full `features.csv` dataset to predict whether the song is popular based on the number of listens in `tracks.csv`. Popularity can be defined as:
- Low (0-33rd percentile)
- Medium (34-66th percentile)
- High (67-100th percentile)

2. Apply PCA to reduce dimensionality of `features.csv` before training the model.

3. Use only a subset of features (e.g., only Chroma STFT and Tonnetz) to see how well they predict popularity.

4. Use LDA to identify latent topics in the audio features and see if these topics correlate with song popularity.


## Code structure

+ code
  + ann
    + build_train.jl -> buildClassANN, trainClassANN, ANNCrossValidation
    + utils_ann.jl -> 
  + mlj_models
    + models.jl -> getSVCModel, getDecisionTreeModel, getkNNModel
    + train_mlj.jl -> mljCrossValidation
  + general
    + utils_general.jl -> holdOut, calculateMinMaxNormalizationParameters, normalizeMinMax!, accuracy, crossvalidation, oneHotEncoding, classifyOutputs
    + train_metrics.jl -> confusionMatrix, printMetrics, printConfusionMatrix
    + model_factory.jl -> modelCrossValidation, trainClassEnsemble, create_tuned_model
    + utils_plot.jl -> draw_results
  
