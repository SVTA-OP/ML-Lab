
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
 
 
def perform_eda(df, dataset_name="dataset", output_dir="./images", font_path="./times.ttf", target_col=None):
 
    
    df = df.rename(columns=lambda c: str(c).strip())
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
 
    
    id_col = None
    for c in df.columns:
        if c.lower() in ("id", "index", "unnamed: 0"):
            id_col = c
            break
 
    if id_col is not None:
        df = df.drop(columns=[id_col])
        print("Dropped id column:", id_col)
 
    
    numeric_cols = list(df.select_dtypes(include="number").columns)
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
 
    
    target_keywords = ["target", "label", "class", "status", "outcome", "species", "diagnosis",
                        "amount", "price", "salary", "income", "score"]
 
    if target_col is None:
        candidates = []
        for c in df.columns:
            if c not in numeric_cols:
                if df[c].nunique() <= 20:
                    candidates.append(c)
 
        for c in candidates:
            if any(word in c.lower() for word in target_keywords):
                target_col = c
                break
 
        if target_col is None and len(candidates) > 0:
            target_col = candidates[-1]
 
        if target_col is None:
            for c in numeric_cols:
                if any(word in c.lower() for word in target_keywords):
                    target_col = c
                    break
 
        if target_col is None:
            for c in numeric_cols:
                if df[c].nunique() <= 10:
                    target_col = c
                    break
 
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
 
    has_target = target_col is not None
 
    
    problem_type = None
    if has_target:
        if df[target_col].dtype == object or str(df[target_col].dtype) == "category":
            problem_type = "classification"
        elif df[target_col].nunique() <= 20:
            problem_type = "classification"
        else:
            problem_type = "regression"
 
    print("Detected target column:", target_col)
    print("Detected problem type:", problem_type)
    print("Numeric feature columns:", numeric_cols)
 
    if len(numeric_cols) == 0:
        print("No numeric columns found. Stopping here.")
        return df
 
    
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()
    else:
        plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 15
 
    os.makedirs(output_dir, exist_ok=True)
 
    
    max_cols = 6
    if len(numeric_cols) > max_cols:
        print("Too many numeric columns (", len(numeric_cols), "), plotting only the first", max_cols)
        cols = numeric_cols[:max_cols]
    else:
        cols = numeric_cols
 
    n_cols = len(cols)
 
    
    
    
    if problem_type == "classification" or not has_target:
 
        n_rows = 3
        fig = plt.figure(figsize=(6 * n_cols, 15))
        gs = fig.add_gridspec(n_rows, n_cols)
 
        
        for i in range(n_cols):
            col = cols[i]
            ax = fig.add_subplot(gs[0, i])
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(col, fontweight="bold")
            ax.set_xlabel(col, fontweight="bold")
            ax.set_ylabel("Count", fontweight="bold")
 
        
        for i in range(n_cols):
            col = cols[i]
            ax = fig.add_subplot(gs[1, i])
            if has_target:
                sns.boxplot(x=target_col, y=col, data=df, ax=ax)
                ax.set_xlabel(target_col, fontweight="bold")
            else:
                sns.boxplot(y=df[col], ax=ax)
                ax.set_xlabel("", fontweight="bold")
            ax.set_title(col, fontweight="bold")
            ax.set_ylabel(col, fontweight="bold")
 
        
        ax_heat = fig.add_subplot(gs[2, :])
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax_heat)
        ax_heat.set_title("Correlation Heatmap", fontweight="bold")
 
        plt.tight_layout()
        combined_path = output_dir + "/" + dataset_name + "_classification_eda.pdf"
        plt.savefig(combined_path, format="pdf", dpi=600)
        plt.show()
        plt.close(fig)
 
    
    
    
    if problem_type == "regression":
 
        n_rows = 3
        fig = plt.figure(figsize=(6 * n_cols, 15))
        gs = fig.add_gridspec(n_rows, n_cols)
 
        
        for i in range(n_cols):
            col = cols[i]
            ax = fig.add_subplot(gs[0, i])
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(col, fontweight="bold")
            ax.set_xlabel(col, fontweight="bold")
            ax.set_ylabel("Count", fontweight="bold")
 
        
        for i in range(n_cols):
            col = cols[i]
            ax = fig.add_subplot(gs[1, i])
            sns.scatterplot(x=col, y=target_col, data=df, ax=ax)
            ax.set_title(col + " vs " + target_col, fontweight="bold")
            ax.set_xlabel(col, fontweight="bold")
            ax.set_ylabel(target_col, fontweight="bold")
 
        
        ax_target_hist = fig.add_subplot(gs[2, 0:max(n_cols // 2, 1)])
        sns.histplot(df[target_col], kde=True, ax=ax_target_hist)
        ax_target_hist.set_title(target_col + " Distribution", fontweight="bold")
        ax_target_hist.set_xlabel(target_col, fontweight="bold")
        ax_target_hist.set_ylabel("Count", fontweight="bold")
 
        half = max(n_cols // 2, 1)
        if half < n_cols:
            ax_heat = fig.add_subplot(gs[2, half:n_cols])
            corr_cols = numeric_cols + [target_col]
            sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", ax=ax_heat)
            ax_heat.set_title("Correlation Heatmap", fontweight="bold")
 
        plt.tight_layout()
        combined_path = output_dir + "/" + dataset_name + "_regression_eda.pdf"
        plt.savefig(combined_path, format="pdf", dpi=600)
        plt.show()
        plt.close(fig)
 
    
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

print("Perform EDA imported")


if __name__ == "__main__":

    df1 = pd.read_csv("Iris.csv")
    perform_eda(df1, dataset_name="iris")

    df2 = pd.read_csv("loan_approval_dataset.csv")
    perform_eda(df2, dataset_name="loan")