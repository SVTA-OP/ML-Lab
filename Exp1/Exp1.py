import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager

df = pd.read_csv("Iris.csv")
# print(df)
print("======================================")
print("Dataset Shape: ",df.shape)
print("\nColumn Types: ",df.dtypes)


# Missing values
print("\nMissing Values: ",df.isnull().sum())

# Overall summary statistics (count, mean, std, min, quartiles, max)
df.describe()

print("======================================")
numeric_cols = df.select_dtypes(include='number').columns.drop('Id')

os.makedirs('./images', exist_ok=True)
font_path = './times.ttf'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['font.size'] = 15

cols = list(numeric_cols)
n_cols = len(cols)

fig = plt.figure(figsize=(24, 30))
gs = fig.add_gridspec(7, n_cols)

for i, col in enumerate(cols):
    ax = fig.add_subplot(gs[0, i])
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(col, fontweight='bold')
    ax.set_xlabel(col, fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')

for i, col in enumerate(cols):
    ax = fig.add_subplot(gs[1, i])
    sns.boxplot(x='Species', y=col, data=df, ax=ax)
    ax.set_title(col, fontweight='bold')
    ax.set_xlabel('Species', fontweight='bold')
    ax.set_ylabel(col, fontweight='bold')

for i, col in enumerate(cols):
    ax = fig.add_subplot(gs[2, i])
    sns.violinplot(x='Species', y=col, data=df, ax=ax)
    ax.set_title(col, fontweight='bold')
    ax.set_xlabel('Species', fontweight='bold')
    ax.set_ylabel(col, fontweight='bold')

for i, col in enumerate(cols):
    ax = fig.add_subplot(gs[3, i])
    sns.swarmplot(x='Species', y=col, data=df, ax=ax)
    ax.set_title(col, fontweight='bold')
    ax.set_xlabel('Species', fontweight='bold')
    ax.set_ylabel(col, fontweight='bold')

ax_scatter1 = fig.add_subplot(gs[4, 0:2])
sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', hue='Species', data=df, ax=ax_scatter1)
ax_scatter1.set_title('PetalLengthCm vs PetalWidthCm', fontweight='bold')
ax_scatter1.set_xlabel('PetalLengthCm', fontweight='bold')
ax_scatter1.set_ylabel('PetalWidthCm', fontweight='bold')

ax_scatter2 = fig.add_subplot(gs[4, 2:4])
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=df, ax=ax_scatter2)
ax_scatter2.set_title('SepalLengthCm vs SepalWidthCm', fontweight='bold')
ax_scatter2.set_xlabel('SepalLengthCm', fontweight='bold')
ax_scatter2.set_ylabel('SepalWidthCm', fontweight='bold')

ax_heat = fig.add_subplot(gs[5, 0:2])
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax_heat)
ax_heat.set_title('Correlation Heatmap', fontweight='bold')

ax_count = fig.add_subplot(gs[5, 2:4])
sns.countplot(x='Species', data=df, ax=ax_count)
ax_count.set_title('Species Count', fontweight='bold')
ax_count.set_xlabel('Species', fontweight='bold')
ax_count.set_ylabel('Count', fontweight='bold')

plt.tight_layout()
plt.savefig('./images/combined_eda.eps', format='eps')
plt.show()

pair = sns.pairplot(df.drop('Id', axis=1), hue='Species')
pair.savefig('./images/pairplot.eps', format='eps')
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