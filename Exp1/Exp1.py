import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
 
 
def perform_eda(df, dataset_name="dataset", output_dir="./images", font_path="./times.ttf", target_col=None):
 
    # ---- Step 0: clean up column names (some CSVs have stray leading/trailing spaces) ----
    df = df.rename(columns=lambda c: c.strip())
    if target_col is not None:
        target_col = target_col.strip()
 
    print("=" * 60)
    print("Dataset:", dataset_name)
    print("Shape:", df.shape)
    print("-" * 60)
    print("Column Types:")
    print(df.dtypes)
    print()
    print("Missing Values:")
    print(df.isnull().sum())
    print()
    print("Summary Statistics:")
    print(df.describe())
    print("=" * 60)
 
    # ---- Step 1: find an id column and drop it ----
    id_col = None
    for c in df.columns:
        if c.lower() in ("id", "index", "unnamed: 0"):
            id_col = c
            break
 
    if id_col is not None:
        df = df.drop(columns=[id_col])
        print("Dropped id column:", id_col)
 
    # ---- Step 2: find numeric columns ----
    numeric_cols = list(df.select_dtypes(include="number").columns)
 
    # ---- Step 3: find the target column ----
    # if the caller passed target_col explicitly, use that and skip auto-detection
    target_keywords = ["target", "label", "class", "status", "outcome", "species", "diagnosis"]
 
    if target_col is None:
        # collect all low-cardinality non-numeric candidates first
        candidates = []
        for c in df.columns:
            if c not in numeric_cols:
                if df[c].nunique() <= 20:
                    candidates.append(c)
 
        # prefer a candidate whose name matches a common "target" keyword
        for c in candidates:
            if any(word in c.lower() for word in target_keywords):
                target_col = c
                break
 
        # otherwise fall back to the LAST low-cardinality candidate
        # (target columns are conventionally placed last in a dataset,
        # unlike other categorical feature columns which tend to come first)
        if target_col is None and len(candidates) > 0:
            target_col = candidates[-1]
 
        # if still nothing, look for a numeric column with few unique values (like Outcome = 0/1)
        if target_col is None:
            for c in numeric_cols:
                if df[c].nunique() <= 10:
                    target_col = c
                    break
 
    # if the target turned out to be numeric, remove it from the numeric feature list
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
 
    has_target = target_col is not None
    print("Detected target column:", target_col)
    print("Numeric feature columns:", numeric_cols)
 
    if len(numeric_cols) == 0:
        print("No numeric columns found. Stopping here.")
        return df
 
    # ---- Step 4: set up font ----
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()
    else:
        plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 15
 
    os.makedirs(output_dir, exist_ok=True)
 
    # ---- Step 5: limit number of columns plotted per row so wide datasets stay readable ----
    max_cols = 6
    if len(numeric_cols) > max_cols:
        print("Too many numeric columns (", len(numeric_cols), "), plotting only the first", max_cols)
        cols = numeric_cols[:max_cols]
    else:
        cols = numeric_cols
 
    n_cols = len(cols)
 
    # ---- Step 6: build the grid of plots ----
    fig = plt.figure(figsize=(6 * n_cols, 30))
    gs = fig.add_gridspec(6, n_cols)
 
    # Row 0: histograms
    for i in range(n_cols):
        col = cols[i]
        ax = fig.add_subplot(gs[0, i])
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel(col, fontweight="bold")
        ax.set_ylabel("Count", fontweight="bold")
 
    # Row 1: boxplots
    if has_target:
        for i in range(n_cols):
            col = cols[i]
            ax = fig.add_subplot(gs[1, i])
            sns.boxplot(x=target_col, y=col, data=df, ax=ax)
            ax.set_title(col, fontweight="bold")
            ax.set_xlabel(target_col, fontweight="bold")
            ax.set_ylabel(col, fontweight="bold")
 
    # Row 2: violin plots
    if has_target:
        for i in range(n_cols):
            col = cols[i]
            ax = fig.add_subplot(gs[2, i])
            sns.violinplot(x=target_col, y=col, data=df, ax=ax)
            ax.set_title(col, fontweight="bold")
            ax.set_xlabel(target_col, fontweight="bold")
            ax.set_ylabel(col, fontweight="bold")
 
    # Row 3: swarm plots
    if has_target:
        for i in range(n_cols):
            col = cols[i]
            ax = fig.add_subplot(gs[3, i])
            sns.swarmplot(x=target_col, y=col, data=df, ax=ax)
            ax.set_title(col, fontweight="bold")
            ax.set_xlabel(target_col, fontweight="bold")
            ax.set_ylabel(col, fontweight="bold")
 
    # Row 4: two scatter plots using the first four numeric columns available
    if n_cols >= 2:
        x1, y1 = cols[0], cols[1]
        ax_scatter1 = fig.add_subplot(gs[4, 0:2])
        if has_target:
            sns.scatterplot(x=x1, y=y1, hue=target_col, data=df, ax=ax_scatter1)
        else:
            sns.scatterplot(x=x1, y=y1, data=df, ax=ax_scatter1)
        ax_scatter1.set_title(x1 + " vs " + y1, fontweight="bold")
        ax_scatter1.set_xlabel(x1, fontweight="bold")
        ax_scatter1.set_ylabel(y1, fontweight="bold")
 
    if n_cols >= 4:
        x2, y2 = cols[2], cols[3]
        ax_scatter2 = fig.add_subplot(gs[4, 2:4])
        if has_target:
            sns.scatterplot(x=x2, y=y2, hue=target_col, data=df, ax=ax_scatter2)
        else:
            sns.scatterplot(x=x2, y=y2, data=df, ax=ax_scatter2)
        ax_scatter2.set_title(x2 + " vs " + y2, fontweight="bold")
        ax_scatter2.set_xlabel(x2, fontweight="bold")
        ax_scatter2.set_ylabel(y2, fontweight="bold")
 
    # Row 5: correlation heatmap and target countplot
    half = n_cols // 2
    if half < 1:
        half = 1
 
    ax_heat = fig.add_subplot(gs[5, 0:half])
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax_heat)
    ax_heat.set_title("Correlation Heatmap", fontweight="bold")
 
    if has_target and half < n_cols:
        ax_count = fig.add_subplot(gs[5, half:n_cols])
        sns.countplot(x=target_col, data=df, ax=ax_count)
        ax_count.set_title(target_col + " Count", fontweight="bold")
        ax_count.set_xlabel(target_col, fontweight="bold")
        ax_count.set_ylabel("Count", fontweight="bold")
 
    plt.tight_layout()
    combined_path = output_dir + "/" + dataset_name + "_combined_eda.eps"
    plt.savefig(combined_path, format="eps", dpi=600)
    plt.show()
    plt.close(fig)
 
    # ---- Step 7: pairplot (only if not too many columns) ----
    if len(numeric_cols) <= 8:
        if has_target:
            plot_df = df[numeric_cols + [target_col]]
            pair = sns.pairplot(plot_df, hue=target_col)
        else:
            plot_df = df[numeric_cols]
            pair = sns.pairplot(plot_df)
        pair_path = output_dir + "/" + dataset_name + "_pairplot.eps"
        pair.savefig(pair_path, format="eps", dpi=600)
        plt.show()
    else:
        print("Skipping pairplot, too many numeric columns:", len(numeric_cols))
 
    # ---- Step 8: descriptive statistics for each numeric column ----
    for col in numeric_cols:
        print("=" * 50)
        print(col)
        print("=" * 50)
        print("Mean:", df[col].mean())
        print("Median:", df[col].median())
        print("Std Dev:", df[col].std())
        print("Min:", df[col].min())
        print("Max:", df[col].max())
        print("Q1:", df[col].quantile(0.25))
        print("Q3:", df[col].quantile(0.75))
        print("Skewness:", df[col].skew())
        print("Kurtosis:", df[col].kurt())
        print()
 
    return df
 


df1 = pd.read_csv("Iris.csv")
perform_eda(df1, dataset_name="iris")
#
df2 = pd.read_csv("loan_approval_dataset.csv")
perform_eda(df2, dataset_name="loan")