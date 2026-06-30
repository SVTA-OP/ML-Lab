import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Iris.csv")
# print(df)

import pandas as pd
import numpy as np

print("======================================")
print("Dataset Shape: ",df.shape)
print("\nColumn Types: ",df.dtypes)


# Missing values
print("\nMissing Values: ",df.isnull().sum())

# Overall summary statistics (count, mean, std, min, quartiles, max)
df.describe()

print("======================================")
numeric_cols = df.select_dtypes(include='number').columns.drop('Id')

for col in numeric_cols:
    sns.histplot(df[col], kde=True)
    plt.title(col)
plt.show()

for col in numeric_cols:
    sns.boxplot(x='Species', y=col, data=df)
    plt.title(col)
plt.show()

for col in numeric_cols:
    sns.violinplot(x='Species', y=col, data=df)
    plt.title(col)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', hue='Species', data=df, ax=axes[0])
axes[0].set_title('PetalLengthCm vs PetalWidthCm')
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=df, ax=axes[1])
axes[1].set_title('SepalLengthCm vs SepalWidthCm')
plt.tight_layout()
plt.show()

sns.pairplot(df.drop('Id', axis=1), hue='Species')
plt.show()

sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.show()

sns.countplot(x='Species', data=df)
plt.show()

for col in numeric_cols:
    sns.swarmplot(x='Species', y=col, data=df)
    plt.title(col)
plt.show()






for col in numeric_cols:
    # print("======================================")
    print("="*50)
    print(col)
    print("="*50)
    print("Mean:", df[col].mean())
    print("Median:", df[col].median())
    print("Std Dev:", df[col].std())
    print("Min:", df[col].min())
    print("Max:", df[col].max())
    print("Q1:", df[col].quantile(0.25))
    print("Q3:", df[col].quantile(0.75))
    print("Skewness:", df[col].skew())
    print("Kurtosis:", df[col].kurt())
    # print("======================================")
    print()