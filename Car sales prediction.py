import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('car_data.csv')

# Preprocessing
# Handle missing values if any
data.dropna(inplace=True)

# Encode categorical variables
data = pd.get_dummies(data, columns=['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem'])

# Features and target variable
X = data.drop(['car_ID', 'CarName', 'price'], axis=1)
y = data['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = XGBRegressor()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
print(f"Root Mean Squared Error: {rmse}")

# Predict new prices
new_data = X_test.iloc[0]  # You can replace this with your new data
predicted_price = model.predict([new_data])
print(f"Predicted Price for New Data: {predicted_price}")
