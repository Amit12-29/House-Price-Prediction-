# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 2: Load Dataset
data = {
    'area': [1000, 1500, 1800, 2400, 3000],
    'bedrooms': [2, 3, 3, 4, 4],
    'price': [3000000, 4500000, 5000000, 6500000, 8000000]
}

df = pd.DataFrame(data)

# Step 3: Features & Target
X = df[['area', 'bedrooms']]
y = df['price']

# Step 4: Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 5: Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Prediction (Fixed)
new_data = pd.DataFrame([[2000, 3]], columns=['area', 'bedrooms'])
prediction = model.predict(new_data)

print("Predicted Price:", prediction[0])