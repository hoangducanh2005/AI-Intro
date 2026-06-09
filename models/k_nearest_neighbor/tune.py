#FIND THE Best value of N
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
from IPython.display import display, HTML
sns.color_palette("Paired")
sns.set_style("whitegrid")

# Load the dataset
df = pd.read_csv('../../dataset/diabetes.csv')

# Calculate the IQR for each column
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Define the outlier detection threshold factor
outlier_threshold_factor = 1.5

# Detect outliers using the IQR method
outliers = ((df < (Q1 - outlier_threshold_factor * IQR)) | (df > (Q3 + outlier_threshold_factor * IQR)))

# Remove rows with outliers
df_no_outliers = df[~outliers.any(axis=1)]
df = df[~outliers.any(axis=1)]

#create numpy arrays for features and target
X = df.drop('Outcome',axis=1).values
y = df['Outcome'].values

# Split the dataset into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=666)

# Splitting the training data into training and validation sets
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

n_values = list(range(1, 30))
accuracies = []

for n in n_values:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train_sub, y_train_sub)
    y_pred_val = knn.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred_val)
    accuracies.append(accuracy)

best_n = n_values[np.argmax(accuracies)]
colored_text = f'<span style="color: #1f78b4; font-weight: bold;">Best value of n: {best_n}</span>'
display(HTML(colored_text))

plt.figure()
plt.plot(n_values, accuracies, marker='o',color = "deeppink")
plt.title("Accuracy vs. Number of Neighbors (n)")
plt.xlabel("Number of Neighbors (n)")
plt.ylabel("Accuracy")
plt.xticks(n_values)
plt.grid(True)
plt.show()

long_text = ''' <span style="color: #1f78b4;"> In the context of applying the K-Nearest Neighbors (KNN) algorithm to our dataset, a thorough evaluation of various values of n (representing the number of nearest neighbors) has been conducted. This evaluation aimed to identify the optimal value that yields the most favorable results in terms of predictive accuracy and generalization. After careful analysis and experimentation, it has been observed that the value of n equal to 9 demonstrates superior performance for our specific dataset. This choice strikes a balance between incorporating sufficient local information from neighboring data points while avoiding overfitting. Therefore, based on our evaluations, we have determined that setting n to 9 optimally aligns with the characteristics of our dataset and contributes to achieving robust and reliable predictions. </span> '''

display(HTML(long_text))