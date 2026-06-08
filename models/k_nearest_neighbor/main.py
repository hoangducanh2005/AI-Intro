import warnings
warnings.filterwarnings("ignore", message="A NumPy version.*")
warnings.filterwarnings("ignore", message="The figure layout.*")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from IPython.display import HTML
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from yellowbrick.classifier import ROCAUC
from matplotlib.colors import ListedColormap
sns.color_palette("Paired")
sns.set_style("whitegrid")

# Load the dataset
df = pd.read_csv('../../dataset/diabetes.csv')

# Check the shape of the DataFrame
shape = df.shape
color_hex = "#1f78b4"
colored_shape_str = "\033[38;2;{};{};{}m{}x{}\033[0m".format(
    int(color_hex[1:3], 16),
    int(color_hex[3:5], 16),
    int(color_hex[5:], 16),
    shape[0],
    shape[1]
)
print("DataFrame shape:", colored_shape_str)

# Calculate the number of unique values in each column
unique_value_counts = df.nunique()

for column, count in unique_value_counts.items():
    colored_count_str = "\033[38;2;{};{};{}m{}\033[0m".format(
        int(color_hex[1:3], 16),
        int(color_hex[3:5], 16),
        int(color_hex[5:], 16),
        count
    )
    print(f"{column}: {colored_count_str} unique values")
