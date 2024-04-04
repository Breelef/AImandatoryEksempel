import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

df = pd.read_csv("cars.csv")
df.rename(columns={'mpg': 'kpl'}, inplace=True)
df['kpl'] = df['kpl'] * 0.4

numeric_columns = df.columns.difference(['car name'])

df = df.apply(pd.to_numeric, errors='coerce')

print(df.dtypes)

df.replace('?', np.nan, inplace=True)

df.fillna(df.mean(), inplace=True)

fig, ax = plt.subplots()
fig.subplots_adjust(hspace=0.8, wspace=0.8, left=0.2, right=1.5)
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
plt.figtext(1.52, 0.5, description, fontsize=10, ha="left")

plt.show()

average_score = df.iloc[:, 0].mean()
print(average_score)


def categorize_kpl(kpl):
    if kpl > average_score:
        return "high"
    else:
        return "low"


# Apply the function to create categorical labels
df['kpl_category'] = df['kpl'].apply(categorize_kpl)

# Separate input features and target variable
X = df.drop(columns=["kpl", "kpl_category", "car name"])  # Drop mpg, mpg_category, and car name
y = df["kpl_category"]  # Target variable is mpg_category

# Perform feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode categorical labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Predict for new data (Cylinders, Displacement,Horsepower, Weight, acceleration, origin)
new_data = [[4, 97, 88, 2130, 14.5, 70, 3]]  # Example new data point

new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
predicted_label = label_encoder.inverse_transform(prediction.astype(int))
print("Predicted kpl Category for new data:", predicted_label)
# 0 = Low (Under 9,4 KPL(average)) 1 = High(Over 9,4 KPL)
