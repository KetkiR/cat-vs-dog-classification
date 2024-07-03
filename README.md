# Cats vs. Dogs Classification Project

## Description
Developed a convolutional neural network (CNN) model to classify images of cats and dogs using a comprehensive data pipeline including data loading, preprocessing, model building, training, and evaluation.

## Techniques Used

### Data Loading and Preparation
- Downloaded and extracted dataset from Kaggle.
- Used `image_dataset_from_directory` to load training and validation datasets.
- Normalized image pixel values for better model performance.

### Model Architecture
- Built a CNN model using `Sequential` API.
- Added convolutional layers with `Conv2D`, followed by `BatchNormalization` and `MaxPooling2D` layers to reduce spatial dimensions.
- Flattened the output from convolutional layers and added fully connected `Dense` layers with `Dropout` for regularization.
- Used `ReLU` activation functions for hidden layers and `sigmoid` activation for the output layer.

### Model Compilation and Training
- Compiled the model with `adam` optimizer and `binary_crossentropy` loss function.
- Included `EarlyStopping` callback to prevent overfitting by monitoring validation loss.
- Trained the model for 10 epochs and validated on the validation dataset.

### Evaluation and Visualization
- Evaluated model performance using accuracy and loss metrics.
- Plotted training and validation accuracy and loss over epochs to visualize model performance.

## Tools and Technologies
- Python
- TensorFlow
- Keras
- Kaggle API
- Matplotlib
- Zipfile

## Project Workflow

1. **Data Loading**
   - Downloaded dataset from Kaggle.
   - Extracted and organized images into training and validation directories.

2. **Data Preprocessing**
   - Normalized image data by scaling pixel values.
   - Used image data generators to load and preprocess images in batches.

3. **Model Building**
   - Constructed a CNN model with multiple convolutional, pooling, and fully connected layers.
   - Applied batch normalization and dropout for regularization.

4. **Model Training**
   - Compiled the model with appropriate loss function and optimizer.
   - Trained the model with early stopping to avoid overfitting.

5. **Model Evaluation**
   - Visualized training and validation accuracy and loss to assess model performance.
