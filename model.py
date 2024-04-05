import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


file_path = 'cars.csv'
data = pd.read_csv(file_path)

data['kpl'] = data['mpg'] * 0.425

data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
data_cleaned = data.dropna(subset=['horsepower', 'weight', 'model year', 'kpl'])
data_cleaned.isnull().sum(), data_cleaned.head()

features = data_cleaned[['weight', 'horsepower', 'model year']]
target = data_cleaned['kpl']

X_train, y_train = train_test_split(features, target, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)


print("Coefficients:", model.coef_)

print("Intercept:", model.intercept_)