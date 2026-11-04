# implementation of linear regression to show how to it work
# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# small dataset used 
data = {
    'area': [1000, 1500, 1800, 2400, 3000],
    'bedrooms': [2, 3, 3, 4, 4],
    'price': [3000000, 4500000, 5000000, 6500000, 8000000]
}

df = pd.DataFrame(data)

# Features & Target extraction
X = df[['area', 'bedrooms']]
y = df['price']

# Step 4: Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#  Model Training of on linearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Prediction (Fixed)
new_data = pd.DataFrame([[2000, 3]], columns=['area', 'bedrooms'])
prediction = model.predict(new_data)

print("Predicted Price:", prediction[0])
