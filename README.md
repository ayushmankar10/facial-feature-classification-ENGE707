# Facial Feature Classification

Individual project for AUT ENGE707 Data Engineering & AI, Assessment 3.

This project develops a deep learning pipeline to classify images of cat and dog facial features into eight classes:

- CatEye
- DogEye
- CatFace
- DogFace
- CatMouth
- DogMouth
- CatNose
- DogNose

## Dataset

The dataset is provided through the ENGE707 Kaggle competition. The labelled training set contains 63,660 images.

| Class | Training images |
|---|---:|
| CatEye | 13,884 |
| DogEye | 11,580 |
| CatFace | 6,942 |
| CatMouth | 6,942 |
| CatNose | 6,942 |
| DogFace | 5,790 |
| DogMouth | 5,790 |
| DogNose | 5,790 |

The `train_labels.csv` file contains two columns:

- `image_id`: the identifier used to match each image to its label.
- `class`: the target class.

Image IDs must retain their leading zeros when matching filenames.

## Project Structure

Place the competition files inside `data/`:

- `data/train/`: training images.
- `data/test/`: unlabelled test images.
- `data/train_labels.csv`: training image IDs and labels.
- `data/sample_submission.csv`: required Kaggle submission format.

Dataset files, Python environments, trained models, and generated predictions are excluded from Git using `.gitignore`.

## Planned Workflow

1. Explore the training data and check image quality, dimensions, class distribution, and duplicates.
2. Create training and validation splits while preventing data leakage.
3. Implement image preprocessing and training augmentation.
4. Train and evaluate a baseline deep learning model.
5. Compare experiments and improve validation performance.
6. Generate test predictions and prepare a Kaggle submission.

## Competition Requirements

- Use only the supplied competition dataset.
- Pre-trained models and standard data augmentation are permitted.
- Use training data for model development and validation.
- Use test images only to generate submission predictions.
- Do not manually label test images.
- Do not share private code, trained models, or prediction files with other students.
- Submit predictions as a CSV with columns `image_id,class`.

## Current Progress

- Created the project repository and `.gitignore`.
- Checked the training labels and class counts.
- Confirmed that the labels contain no missing values or duplicate image IDs.

Environment setup, image inspection, model training, and evaluation are still to be completed.

## Reproducibility

Setup instructions, dependencies, and execution commands will be added as the pipeline is implemented. Experiment settings and results will be recorded throughout development.
