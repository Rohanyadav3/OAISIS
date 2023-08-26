import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the sales data (you'll need to prepare and provide the dataset)
data = pd.read_csv('sales_data_sample.csv')

# Assume the dataset has 'features' and 'sales' columns
X = data[['feature1', 'feature2', ...]]  # Features
y = data['sales']  # Sales

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)