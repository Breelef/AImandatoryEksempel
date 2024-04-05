import pandas as pd

df = pd.read_csv("cars.csv")
df.rename(columns={'mpg': 'kpl'}, inplace=True)
df['kpl'] = df['kpl'] * 0.425

numeric_columns = df.columns.difference(['car name'])

df = df.apply(pd.to_numeric, errors='coerce')

average_cylinders = df.iloc[:, 1].mean()
average_displacement = df.iloc[:, 2].mean()
average_horsepower = df.iloc[:, 3].mean()
average_weight = df.iloc[:, 4].mean()
average_acceleration = df.iloc[:, 5].mean()

print(f"average cylinders: {average_cylinders}") #5,45
print(f"average displacement: {average_displacement}") #193,42
print(f"average horsepower: {average_horsepower}") #104,46
print(f"average weight: {average_weight}") #2970,42
print(f"average acceleration: {average_acceleration}") #15,56

#8
#320
#144
#3360
#16


#7/8 of each value (times with 0,875)
#8 = 7
#320 = 280
#144 = 126
#3360 = 2940
#16 = 14