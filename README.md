# Insurance Cost Prediction with TensorFlow and Keras

## Project Overview

This project demonstrates a machine learning pipeline for predicting insurance costs using a neural network built with TensorFlow and Keras. The project covers data preprocessing, model design, training, and evaluation.

## Dataset

The dataset used in this project is the **Insurance** dataset, which contains information about:
- Age
- Sex
- BMI
- Children
- Smoker
- Region
- Charges (insurance cost)

## Project Steps

1. **Data Preprocessing:**
   - Load the dataset and inspect the features.
   - Handle categorical variables with one-hot encoding.
   - Standardize numerical features.

2. **Model Design:**
   - Build a neural network with an input layer, a hidden layer with 128 neurons, and an output layer.
   - Compile the model using the Adam optimizer and mean squared error loss.

3. **Model Training:**
   - Train the model using 40 epochs and a batch size of 1.
   - Evaluate the model on the test data and report the mean absolute error (MAE).

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/simk1/insurance-cost-prediction.git
   cd insurance-cost-prediction
   
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
   
## Results
The model achieves a mean absolute error (MAE) of 2442 on the test set.

## Conclusion
This project provides a basic implementation of a neural network for regression tasks. It can be extended with more advanced preprocessing, hyperparameter tuning, and additional feature engineering.
