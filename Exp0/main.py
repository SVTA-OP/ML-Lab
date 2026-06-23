# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Set the path to the file you'd like to load
file_path = "Cancer_Data.csv"

# Load the latest version
''' df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "erdemtaha/cancer-data",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)
print("First 5 records:", df.head())

df.to_csv("Dataset.csv") '''

df = pd.read_csv("Dataset.csv")

df = df.fillna(0)

x = df.drop(columns=["id", "diagnosis"])

y = df["diagnosis"]


x_train, x_test, y_train, y_test = train_test_split(x, y)

model = LogisticRegression(max_iter=1000)

# 4. Train the model
model.fit(x_train, y_train)

# 5. Make predictions
predictions = model.predict(pd.DataFrame({
    'radius_mean': [18.5],
    'texture_mean': [22.1],
    'perimeter_mean': [123.4],
    'area_mean': [1025.0],
    'smoothness_mean': [0.11],
    'compactness_mean': [0.22],
    'concavity_mean': [0.28],
    'concave points_mean': [0.13],
    'symmetry_mean': [0.21],
    'fractal_dimension_mean': [0.07],
    'radius_se': [0.75],
    'texture_se': [1.10],
    'perimeter_se': [4.80],
    'area_se': [95.0],
    'smoothness_se': [0.006],
    'compactness_se': [0.04],
    'concavity_se': [0.05],
    'concave points_se': [0.02],
    'symmetry_se': [0.02],
    'fractal_dimension_se': [0.005],
    'radius_worst': [24.0],
    'texture_worst': [28.5],
    'perimeter_worst': [160.0],
    'area_worst': [1850.0],
    'smoothness_worst': [0.15],
    'compactness_worst': [0.52],
    'concavity_worst': [0.61],
    'concave points_worst': [0.23],
    'symmetry_worst': [0.38],
    'fractal_dimension_worst': [0.11]
}))
print(predictions)

