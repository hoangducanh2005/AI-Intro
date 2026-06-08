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