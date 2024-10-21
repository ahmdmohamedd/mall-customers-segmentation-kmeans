s# Mall Customers Segmentation using K-Means Clustering

## Project Overview
This project implements a customer segmentation system using **K-Means clustering** from scratch in **MATLAB**. The goal is to categorize mall customers based on their **Age**, **Annual Income**, and **Spending Score** into distinct groups (clusters) to identify patterns and trends in customer behavior. The optimal number of clusters is determined using the **Elbow Method**.

## Dataset
The dataset used is the `Mall_Customers.csv` file, which contains the following attributes for each customer:
- **CustomerID**: A unique identifier for each customer.
- **Genre**: Gender of the customer (Male/Female).
- **Age**: Age of the customer in years.
- **Annual Income (k$)**: Annual income of the customer in thousand dollars.
- **Spending Score (1-100)**: A score assigned by the mall based on the customer’s behavior and spending nature.

The features **Age**, **Annual Income**, and **Spending Score** are used to perform clustering.

## Files in the Repository
- `kmeans.m`: MATLAB script that implements the K-Means clustering algorithm from scratch, along with the Elbow Method to automatically determine the optimal number of clusters.
- **Saved Visualizations**:
  - `Elbow Method.png`: A plot of the elbow method used to determine the optimal number of clusters.
  - `Customer Segmentation using K-means.png`: Visualization of the final customer segmentation based on the optimal number of clusters.
  - `Silhouette Plot.png`: A silhouette plot to evaluate the consistency of clusters.

## Methodology
1. **Data Preprocessing**: 
   - The dataset is read and preprocessed, ensuring that the correct features (`Age`, `Annual Income`, and `Spending Score`) are extracted for clustering.
   
2. **K-Means Clustering from Scratch**:
   - Random initialization of centroids.
   - Iterative process to assign each data point to the nearest centroid.
   - Recomputing centroids based on the mean of data points in each cluster.
   - The process repeats until centroids stabilize or reach a predefined number of iterations.

3. **Optimal Number of Clusters (Elbow Method)**:
   - The Elbow Method is applied by calculating the sum of squared errors (inertia) for a range of cluster numbers (1 to 10). 
   - The point at which the reduction in inertia begins to slow down significantly (the "elbow") is identified as the optimal number of clusters.
   - The number of clusters (`k`) is then automated from the elbow method.

4. **Evaluation**:
   - A **Silhouette Plot** is used to evaluate the consistency of clusters.
   - Visualizations of customer segmentation help interpret the clusters based on features like age and spending score.

## Results
- **Optimal Number of Clusters**: The elbow method determined the optimal number of clusters as **5**.
- **Cluster Segmentation**: Based on clustering, customers are divided into distinct segments that show unique behavior patterns in terms of age, income, and spending score.
  
### Visualizations
- **Elbow Method**: Used to find the optimal number of clusters.
- **Customer Segmentation**: A 2D scatter plot showing the clusters based on selected features.
- **Silhouette Plot**: Evaluates the cluster cohesion and separation.

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/mall-customers-segmentation-kmeans.git
   ```
2. Open `kmeans.m` in MATLAB.
3. Run the script to see the clustering process and generate the visualizations.
   
Ensure that you have the dataset `Mall_Customers.csv` in the same directory as the script.

## Conclusion
This project demonstrates how to implement K-Means clustering from scratch and apply it to real-world data for customer segmentation. By using the elbow method and evaluating the cluster quality through silhouette scores, meaningful insights into customer behavior are obtained, which can be valuable for businesses to target specific customer groups more effectively.
