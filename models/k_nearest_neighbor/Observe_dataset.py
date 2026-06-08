# Load the dataset
df = pd.read_csv('../../dataset/diabetes.csv')
print(df.head())#Print the first 5 rows of the dataframe.

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
    
# Calculate the IQR for each column
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Define the outlier detection threshold factor
outlier_threshold_factor = 1.5

# Detect outliers using the IQR method
outliers = ((df < (Q1 - outlier_threshold_factor * IQR)) | (df > (Q3 + outlier_threshold_factor * IQR)))

# Display columns with outliers
columns_with_outliers = outliers.any()
print("\033[38;2;238;18;137m"+"Columns with outliers:"+"\033[0m")
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))
prGreen(columns_with_outliers)