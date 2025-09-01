# Real-Life Violence Situations Dataset - Violence Detection

This project demonstrates a violence detection model using a Bidirectional LSTM with a MobileNetV2 base on the Real-Life Violence Situations Dataset.

## Overview

The goal of this project is to build a model that classifies video sequences as either **"Violence"** or **"NonViolence"**. The main steps include:

- **Data Loading and Preparation:** Import dataset, extract frames from videos, and preprocess them.
- **Model Architecture:** Use a Bidirectional LSTM model with MobileNetV2 for feature extraction.
- **Training and Evaluation:** Train the model and evaluate performance with accuracy, confusion matrix, and classification report.
- **Prediction:** Implement functions to predict violence in individual videos and visualize frame-by-frame predictions.

## Dataset

The dataset used is the **Real-Life Violence Situations Dataset** from Kaggle, containing video clips categorized as **"Violence"** and **"NonViolence"**.

## Dependencies

This project requires the following libraries:

- kagglehub
- os
- shutil
- cv2
- math
- random
- numpy
- datetime
- tensorflow
- keras
- collections (deque)
- matplotlib
- sklearn (train_test_split, accuracy_score, confusion_matrix, classification_report)
- seaborn

## Usage

1. **Import Data:** Run the first cell to import the dataset from Kaggle.
2. **Install Dependencies:** Make sure all required libraries are installed.
3. **Run Notebook:** Execute the code cells sequentially for data preprocessing, model training, evaluation, and prediction.

## Code Structure

- **Data Loading and Preprocessing:** Functions to extract frames from videos and create the dataset.
- **Model Definition:** Bidirectional LSTM model definition with MobileNetV2 as the base.
- **Training and Evaluation:** Code to compile, train, and evaluate the model.
- **Prediction Functions:** Functions for violence prediction in videos and displaying results.

## Results

Includes visualization of training history (loss and accuracy), a confusion matrix, and a classification report to assess the model's performance.

## Prediction Examples

Examples demonstrate predictions on both **"Violence"** and **"NonViolence"** video clips, including displaying the original video and sampled frames from the predicted output video (saved to disk).
