import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score

df = pd.read_csv("cars.csv")
df.rename(columns={'mpg': 'kpl'}, inplace=True)
df['kpl'] = df['kpl'] * 0.425

numeric_columns = df.columns.difference(['car name'])

df = df.apply(pd.to_numeric, errors='coerce')

print(df.dtypes)

df.replace('?', np.nan, inplace=True)

df.fillna(df.mean(), inplace=True)

fig, ax = plt.subplots(figsize=(14, 4))  # Adjust the figsize parameter as needed

plt.xticks([], [])
plt.yticks([], [])

for idx in range(6):
    plt.subplot(1, 6, idx + 1)
    origin_df = df.groupby("origin")[list(df.columns[0:6])[idx]].mean()
    sns.barplot(x=origin_df.index, y=origin_df.values)
    plt.title(list(df.columns[0:6])[idx])

description = (
    "1: Made in USA\n"
    "2: Made in Europe\n"
    "3: Made in Asia"
)
plt.figtext(0.9, 0.5, description, fontsize=10, ha="left")

plt.show()

average_score = df.iloc[:, 0].mean()



def categorize_kpl(kpl):
    if kpl > average_score:
        return f"which is lower than average fuel consumption of {average_score:.2f} KPL and/or {average_score / 0.425:.2f} MPG"
    else:
        return f"which is higher than average fuel consumption of {average_score:.2f} KPL and/or {average_score / 0.425:.2f} MPG"


X = df.drop(columns=["kpl", "car name"])
y = df["kpl"] 


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(7, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

adam = Adam(learning_rate=0.008)

model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mse'])


model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

y_pred = model.predict(X_test)



loss, mse = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test MSE:", mse)
r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)
print("Accuracy:" r2*100)

#----------------------------PREDICTION CAR-------------------------
cylinders = 8
displacement = 307
horsepower = 130
weight = 3504
acceleration = 12
model_year = 70
origin = "USA"
number_origin = 0

match origin:
            case "USA": number_origin = 1
            case "EUROPE": number_origin = 2
            case "ASIA": number_origin = 3

new_data = [[cylinders, displacement, horsepower, weight, acceleration, model_year, number_origin]]
#----------------------------PREDICTION CAR-------------------------

new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print("Predicted kpl for new data:", prediction[0][0])

class Car:
    def __init__(self, kpl, cylinders, displacement, horsepower, weight, acceleration, model_year, origin):
        self.kpl = kpl
        self.cylinders = cylinders
        self.displacement = displacement
        self.horsepower = horsepower
        self.weight = weight
        self.acceleration = acceleration
        self.model_year = model_year
        match origin:
            case 1: self.origin = "USA"
            case 2: self.origin = "EUROPE"
            case 3: self.origin = "ASIA"

car = Car(kpl=prediction[0][0], cylinders=new_data[0][0], displacement=new_data[0][1], 
          horsepower=new_data[0][2], weight=new_data[0][3], acceleration=new_data[0][4], 
          model_year=new_data[0][5], origin=new_data[0][6])

print(f"Predict car: \n\n"
      f"Cylinders: {car.cylinders}\n"
      f"Displacement: {car.displacement}\n"
      f"Horsepower: {car.horsepower}\n"
      f"Weight: {car.weight}\n"
      f"Acceleration: {car.acceleration}\n"
      f"Model year: {car.model_year}\n"
      f"Origin: {car.origin}\n"
      f"Predicted Kpl: {car.kpl:.2f}, and/or {car.kpl / 0.425:.2f} in mpg {categorize_kpl(car.kpl)}")
