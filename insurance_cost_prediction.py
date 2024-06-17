import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(35)  # for the reproducibility of results

def design_model(features):
    model = Sequential(name="my_first_model")
    # Input layer
    model.add(InputLayer(shape=(features.shape[1],)))
    # Hidden layer
    model.add(Dense(128, activation='relu'))
    # Output layer
    model.add(Dense(1))
    # Compile the model
    model.compile(loss='mse', metrics=['mae'], optimizer=Adam(learning_rate=0.1))
    return model

# Load the dataset
dataset = pd.read_csv('insurance.csv')

features = dataset.iloc[:, 0:6]
labels = dataset.iloc[:, -1]

# One-hot encoding for categorical variables
features = pd.get_dummies(features)
# Split the data into training and test data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42)

features_train = features_train.astype(float)
features_test = features_test.astype(float)
labels_train = labels_train.astype(float)
labels_test = labels_test.astype(float)

# Standardize numerical features
ct = ColumnTransformer([('standardize', StandardScaler(), ['age', 'bmi', 'children'])], remainder='passthrough')
features_train = ct.fit_transform(features_train)
features_test = ct.transform(features_test)

# Design the model
model = design_model(features_train)
print(model.summary())

# Fit the model using 40 epochs and batch size 1
model.fit(features_train, labels_train, epochs=40, batch_size=1, verbose=1)
# Evaluate the model on the test data
val_mse, val_mae = model.evaluate(features_test, labels_test, verbose=0)

print("MAE: ", val_mae)
print("MSE: ", val_mse)

