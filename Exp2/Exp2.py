import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Exp1.Exp1 import perform_eda
from sklearn.model_selection import train_test_split
from sklearn.neighbours 

df = pd.read_csv("spambase_csv.csv")
# print(df['class'])
perform_eda(df, dataset_name="Spam Email Classification")

