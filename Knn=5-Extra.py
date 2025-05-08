import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import euclidean

# Step 1: Create the dataset
data = {
    'Age': [20, 25, 30, 35, 45, 50],
    'Income': [50, 75, 80, 150, 200, 250],
    'Credit_History': [0, 2, 2, 4, 7, 9],
    'Loan_Repaid': ['No', 'No', 'Yes', 'Yes', 'Yes', 'Yes']
}
df = pd.DataFrame(data)

# Step 2: Add new customer (to be predicted)
new_customer = pd.DataFrame({'Age': [29], 'Income': [95], 'Credit_History': [3], 'Loan_Repaid': ['?']})
df = pd.concat([df, new_customer], ignore_index=True)

# Step 3: Normalize features
features = ['Age', 'Income', 'Credit_History']
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(df[features])
normalized_df = pd.DataFrame(normalized_features, columns=features)

# Step 4: Prepare training and test data
X_train = normalized_df.iloc[:-1]  # Exclude new customer
y_train = df['Loan_Repaid'][:-1]
new_customer_normalized = normalized_df.iloc[[-1]]

# Step 5: Train KNN with k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
prediction = knn.predict(new_customer_normalized)

# Step 6: Find the 5 nearest neighbors
distances = X_train.apply(lambda row: euclidean(row, new_customer_normalized.iloc[0]), axis=1)
sorted_indices = distances.nsmallest(5).index
nearest_neighbors = df.iloc[sorted_indices][['Age', 'Income', 'Credit_History', 'Loan_Repaid']]

# Print prediction and neighbor votes
print("Prediction for the new customer:", prediction[0])
print("\n5 Nearest Neighbors:\n", nearest_neighbors)
print("\nVote Count:\n", nearest_neighbors['Loan_Repaid'].value_counts())

# Step 7: 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Color map
color_map = {'Yes': 'green', 'No': 'red', '?': 'blue'}
colors = df['Loan_Repaid'].map(color_map)

# Plot existing customers
ax.scatter(
    normalized_df['Age'][:-1],
    normalized_df['Income'][:-1],
    normalized_df['Credit_History'][:-1],
    c=colors[:-1],
    s=60,
    label='Customers'
)

# Plot new customer
ax.scatter(
    new_customer_normalized['Age'],
    new_customer_normalized['Income'],
    new_customer_normalized['Credit_History'],
    color='blue',
    s=100,
    edgecolors='black',
    marker='X',
    label='New Customer'
)

# Axes labels
ax.set_xlabel('Age (normalized)')
ax.set_ylabel('Income (normalized)')
ax.set_zlabel('Credit History (normalized)')
ax.set_title('KNN with k=5: Loan Repayment Prediction')
ax.legend()

plt.show()
