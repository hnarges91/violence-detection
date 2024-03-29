
# Violence Detection Using Pre-trained Models

This project aims to detect violence in videos using pre-trained deep learning models. It provides a pipeline for feature extraction, classification, and evaluation.

## Features

- **Feature Extraction**: Extracts features from video frames using pre-trained convolutional neural network models such as VGG16, VGG19, and ResNet50.
- **Classification**: Utilizes a fully connected neural network for violence classification based on the extracted features.
- **Evaluation**: Provides metrics such as accuracy, loss, classification report, and confusion matrix for evaluating model performance.

## Usage

1. **Setup Environment**: Ensure all dependencies are installed by running `pip install -r requirements.txt`.
2. **Data Preparation**: Organize video data into appropriate directories according to the dataset structure.
3. **Feature Extraction**: Run the `violence_detection` function specifying the pre-trained model, dataset, and pooling type.
4. **Evaluation**: Evaluate the model's performance by examining the classification report, confusion matrix, and plots for accuracy and loss.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- scikit-learn
- matplotlib

## Example

```python

model_name = 'vgg16'
dataset_name = 'example_dataset'
pooling_type = 'mean'

accuracy, loss = violence_detection(model_name, dataset_name, pooling_type)
print("Accuracy:", accuracy)
print("Loss:", loss)
```
## Datasets
The used datasets besides related features can be found at [Google Drive](https://drive.google.com/drive/folders/1sKApYFk1OViZZilpzvQlkzHNiePPhS47?usp=sharing).
