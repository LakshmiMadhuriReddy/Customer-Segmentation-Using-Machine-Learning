import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Create a sample customer dataset
data = {
    "CustomerID": list(range(1, 16)),
    "Age": [19, 21, 20, 23, 31, 35, 40, 42, 36, 29, 50, 55, 48, 60, 45],
    "AnnualIncome": [15, 18, 17, 20, 45, 48, 50, 55, 42, 40, 75, 80, 72, 85, 78],
    "SpendingScore": [85, 81, 77, 90, 40, 45, 50, 42, 48, 55, 20, 15, 25, 10, 30]
}

df = pd.DataFrame(data)

print("CUSTOMER SEGMENTATION PROJECT")
print("\nFirst five customer records:")
print(df.head())

# Step 2: Select features for clustering
features = df[["AnnualIncome", "SpendingScore"]]

# Step 3: Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 4: Train the K-Means model
model = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(scaled_features)

# Step 5: Evaluate the clusters
score = silhouette_score(scaled_features, df["Cluster"])

print("\nCustomer group summary:")
print(df.groupby("Cluster")[["AnnualIncome", "SpendingScore"]].mean().round(2))

print(f"\nSilhouette Score: {score:.3f}")
print("\nCustomer assignments:")
print(df[["CustomerID", "AnnualIncome", "SpendingScore", "Cluster"]])

# Step 6: Visualize the customer groups
plt.figure(figsize=(8, 5))

for cluster in sorted(df["Cluster"].unique()):
    group = df[df["Cluster"] == cluster]
    plt.scatter(
        group["AnnualIncome"],
        group["SpendingScore"],
        label=f"Cluster {cluster}",
        s=80
    )

plt.title("Customer Segmentation Using K-Means")
plt.xlabel("Annual Income (in thousands)")
plt.ylabel("Spending Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("customer_segments.png", dpi=300)
plt.show()

# Step 7: Save the results
df.to_csv("customer_segments_results.csv", index=False)
print("\nResults saved to customer_segments_results.csv")
print("Graph saved to customer_segments.png")
