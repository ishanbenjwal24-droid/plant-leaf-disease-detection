# Plant Leaf Disease Detection Using Machine Learning and Deep Learning

> **Submission note:** Fill every bracketed item after preparing the dataset and running the experiments. Do not report unmeasured results. This project is an educational comparison, not a plant diagnosis system.

## Abstract

This project compares a traditional image-classification pipeline with a convolutional neural network for assigning plant leaf images to PlantVillage dataset classes. The traditional pipeline extracts Histogram of Oriented Gradients (HOG) and HSV color statistics, then uses a Linear Support Vector Classifier. The deep-learning pipeline trains a CNN directly on resized RGB images. Both models use the same reproducible train, validation, and test split. A local web application provides image upload, side-by-side predictions, measured model metrics, and prediction-history metadata.

After running the experiments, summarize the measured results here: **[insert test-set accuracy, macro F1, latency, and model-size comparison]**.

## 1. Introduction and motivation

Plant leaf appearance can provide visual clues about plant health. This project studies how two image-classification approaches learn labels from a public, curated dataset. It also demonstrates the complete path from dataset preparation and training to a local web interface.

## 2. Objectives

1. Prepare a reproducible, class-stratified dataset split and report invalid files and duplicates.
2. Train a HOG/HSV + LinearSVC machine-learning baseline.
3. Train a CNN with augmentation applied only to training data.
4. Compare both models on the same held-out test data using class-aware metrics.
5. Build a local React frontend, FastAPI backend, and SQLite metadata history.

## 3. Dataset

The project uses the PlantVillage dataset. Record the exact source/version used, folder layout, class count, and counts after preparation.

| Item | Measured value |
|---|---|
| Dataset source/version | [fill in] |
| Number of classes retained | [fill in] |
| Unique readable images selected | [fill in] |
| Exact duplicate files excluded | [fill in] |
| Unreadable files | [fill in] |
| Sampling cap per class | [fill in] |
| Train / validation / test counts | [fill in] |

The preparation script hashes image files and excludes exact duplicate groups before splitting. Classes with fewer than three unique readable images are skipped. Class counts can differ slightly for very small classes because the split guarantees at least one sample for train, validation, and test.

## 4. System design

The React frontend submits images to FastAPI. The backend validates the upload, applies EXIF orientation and RGB conversion, dispatches inference to the selected trained model, and stores only result metadata in SQLite. Training outputs are stored locally under `artifacts/`; raw data and large outputs are excluded from Git.

See [architecture diagram](architecture.mmd), [API examples](api_examples.md), and [database schema](database_schema.md).

## 5. Methodology

### 5.1 Traditional ML model

Images are EXIF-corrected, converted to RGB, and resized to 96 × 96 pixels. HOG captures local shape/edge structure; HSV channel means and standard deviations summarize color. A class-balanced LinearSVC is fitted on the resulting feature vectors. Its decision values are presented as model scores, not probabilities.

### 5.2 Deep-learning model

Images are EXIF-corrected, converted to RGB, resized to 128 × 128 pixels, and scaled to [0, 1]. The CNN contains three convolution/max-pooling blocks, global average pooling, dropout, and a softmax output layer. Random flip, rotation, and zoom layers operate only during training. Early stopping uses validation loss.

### 5.3 Split and reproducibility

Both models read `data/splits.csv`, generated with seed 42 and an approximate 70/15/15 train/validation/test allocation. The test set remains untouched during training and tuning. Record the package versions, machine/CPU/GPU, seed, image cap, and training epoch count used for final results.

## 6. Evaluation

Populate this table from `artifacts/ml_metrics.json` and `artifacts/cnn_metrics.json` after training:

| Metric | HOG + LinearSVC | CNN |
|---|---:|---:|
| Test accuracy | [fill in] | [fill in] |
| Macro precision | [fill in] | [fill in] |
| Macro recall | [fill in] | [fill in] |
| Macro F1 | [fill in] | [fill in] |
| Average model inference (ms/image) | [fill in] | [fill in] |
| Model size (bytes) | [fill in] | [fill in] |

Include the confusion matrices and discuss classes with weaker recall or frequent confusion. Do not treat accuracy alone as sufficient where class performance varies.

### Discussion prompts

- Which model produced the stronger macro F1 on this test split?
- Did one model use less storage or provide lower inference latency?
- Which classes were most often confused, and what visual similarities may explain this?
- How did the selected sample cap and image split affect experiment time and interpretation?

## 7. Limitations and responsible use

PlantVillage images were collected in controlled conditions. Backgrounds, lighting, camera angle, crop varieties, and disease severity in field images may differ. Dataset labels and test metrics do not establish real-world diagnostic accuracy. Model outputs are educational predictions and should not replace expert advice. The app does not retain uploaded images.

## 8. Conclusion

After running the experiments, summarize the comparison using measured values and explain the strongest trade-offs: **[fill in conclusion]**. Possible future work includes evaluation on an independent field-image dataset, calibrated uncertainty, and testing on images captured under varied conditions.

## References

1. PlantVillage dataset repository and associated paper: https://github.com/spMohanty/PlantVillage-Dataset
2. TensorFlow/Keras documentation: https://www.tensorflow.org/guide/keras
3. scikit-learn LinearSVC documentation: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
