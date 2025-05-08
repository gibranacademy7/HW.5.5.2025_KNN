"""
Explanation:

We used the K-Nearest Neighbors (KNN) algorithm with k = 3.
That means the algorithm looks for the 3 closest (most similar) customers to the new one,
based on Euclidean distance between the features (Age, Income, Credit History) after normalizing the data.

The three nearest neighbors are:

The new customer himself (unknown label — we ignore this)

A customer who repaid the loan

A customer who did not repay the loan

Out of the 2 known nearest neighbors:

One repaid the loan ("Yes")

One did not ("No")

👁 In case of a tie, sklearn's KNN uses the label of the closest valid neighbor —
which in this case is "Yes".
—
Therefore, the model predicts that the new customer will repay the loan.
"""