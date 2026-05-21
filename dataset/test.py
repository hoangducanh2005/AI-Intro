import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# đọc dữ liệu
df = pd.read_csv("diabetes.csv")

# tách feature và label
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2, # 20% test, 80% train
    random_state=42
)

# Test train + test
# print(X_train.shape)  # trả về số dòng (samples) + số cột (feature)
# print(X_test.shape)

# Bắt đầu Standard Scaler
scaler = StandardScaler()
# scale train
X_train_scaled = scaler.fit_transform(X_train)
# scale test CHỈ transform, KHÔNG fit tránh leakage
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled[:5])