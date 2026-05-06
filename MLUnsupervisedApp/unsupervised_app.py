# Project 4
import streamlit as st
import pandas as pd

# Code for colored boxes to highlight important information and notes.
def colored_box(text, color="#2c5572"):
    st.markdown(
        f"""
        <div style="
            background-color: {color};
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 10px;
        ">
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )

# Introduce users to the app
st.title("Unsupervised Machine Learning App")
st.write("Welcome to the Unsupervised Machine Learning App! This app will allow you to explore the basics of unupervised machine learning with your own custom dataset.")
st.header("What does this app do specifically?")
# Outline what the app will do specifically
st.write("1. Explore three different unsupervised machine learning techniques: k-means clustering, hierarchical clustering, and principal component analysis (PCA).")
st.write("2. Experiment with hyperparameters to see how they impact model results.")
st.write("3. Visualize the results with elbow plots, dendrograms, scatter plots, and silhouette plots to evaluate model performance.")

st.divider()

# Uploading their dataset
st.header("Uploading Your Dataset")
st.write("To begin, please upload your dataset in a CSV format. Please ensure that the data you have chosen contains numeric data so that the models can run effectively.")
colored_box("NOTE: If you do not want to upload your own data or do not have a dataset available, you can choose from three datasets that will be provided below the upload option.", color="#628ba9")



# Allow the user to upload a CSV file of their own.
uploaded_file = st.file_uploader("Choose a CSV file for this app", type="csv")
if uploaded_file is not None:
    # Plug the uploaded CSV file into a dataframe
    df = pd.read_csv(uploaded_file)
    st.write("Your dataset was successfully loaded! Here's a preview:")
    st.dataframe(df.head())
# If they do not upload a file of their own, they can select from the three sample datasets in the folder
else:
    st.write("No file has been uploaded yet. If you do not have your own dataset, please select one of the following sample datasets:")
    dataset_options = ["Social Media Impacts Dataset", "World Happiness Dataset", "Teen Mental Health Dataset"]
    selected_dataset = st.selectbox("Select a dataset", dataset_options)
    
    if selected_dataset == "Social Media Impacts Dataset":
        df = pd.read_csv("sample_data/Social_Media.csv")
        st.write("You have selected the Social Media Impacts Dataset. This is a dataset that records social media usage and patterns associated with high and low users. Here's a preview:")
        st.dataframe(df.head())
        
    elif selected_dataset == "World Happiness Dataset":
        df = pd.read_csv("sample_data/world_happiness_2026.csv")
        st.write("You have selected the World Happiness Dataset. This is a dataset that records the happiness levels of individuals across different countries in 2026. Here's a preview:")
        st.dataframe(df.head())
        
    elif selected_dataset == "Teen Mental Health Dataset":
        df = pd.read_csv("sample_data/Teen_Mental_Health_Dataset.csv")
        st.write("You have selected the Teen Mental Health Dataset. This is a dataset that records the mental health status of teenagers and their usage of Instagram and/or Tiktok. Here's a preview:")
        st.dataframe(df.head())

st.subheader("Dont forget to clean!")
# Make sure to filter for numeric columns or convert categorical variables into dummy variables and drop the rows with missing values. We are cleaning the data here
categorical_columns = df.select_dtypes(exclude=['number']).columns.tolist()
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first = True)
df_encoded = df_encoded.dropna()

st.write("The dataset has been cleaned by dropping rows with missing values and converting categorical variables into dummy variables. Here's a preview of the cleaned dataset:")
st.dataframe(df_encoded.head())
# Only use the first 500 rows of the dataset to ensure that the models run efficiently. 
if len(df_encoded) > 500:
    df_encoded = df_encoded.head(500)
# Scale the data. We want all of our units to be standardized so that no one variable dominates the clustering process.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded)

st.divider()

# Now, give the user the option to choose the technique they want to explore first.
st.header("Choose an Unsupervised Machine Learning Technique")
technique_options = ["K-Means Clustering", "Hierarchical Clustering", "Principal Component Analysis (PCA)"]
selected_technique = st.selectbox("Select a technique to explore", technique_options)

if selected_technique == "K-Means Clustering":
    st.write("Great! You selected K-Means Clustering. This technique groups similar data points into k number of clusters based on their features.")
    with st.expander("How it works (in more detail):"):
        st.markdown('<div style="background-color: #dac17f; padding: 10px; border-radius: 5px;">K-means clustering begins by randomly placing a certain number of centroids (k) down. Each subsequent data point is then assigned to the nearest centroid, forming clusters. The centroids are then recalculated as the mean of the data points in each cluster. This process is repeated until the centroids stabilize (meaning that the means no longer change significantly.)</div>', unsafe_allow_html=True)
    st.write("To get started with this technique, please select the number of clusters (k) you would like to use.")
    k = st.slider("Select the number of clusters (k)", min_value=2, max_value=10, value=5)
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    
    st.write(f"K-Means Clustering has been applied with {k} clusters, k={k}. Here are the cluster centers displayed:")
    st.dataframe(kmeans.cluster_centers_)
    df_results = df_encoded.copy()
    df_results["Cluster"] = kmeans.labels_
    # Display which observations are going to which cluster
    st.subheader("Cluster Assignments")
    st.write("Here is a preview of each observation and the cluster it belongs to (cluster column is all the way to the right):")
    st.dataframe(df_results.head(25))
    # Let's look at the elbow plot to determine which number of clusters is best.
    import matplotlib.pyplot as plt
    st.subheader("Evaluating K-Means Clustering with Elbow Plot and Silhouette Scores")
    inertia = []
    k_values = range(1, 11)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_scaled)
        inertia.append(kmeans.inertia_)
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Plot for K-Means Clustering')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    st.pyplot(plt)
    
# Inertia is the sum of squared distances between each point and its assigned cluster center.

    st.write("In an elbow plot, we typically look for the point on the graph where the y-axis (inertia) starts to level off as the number of clusters increases. It often looks like a kink or 'elbow' in the graph. This is the point referred to as the 'elbow', and it can help us determine the optimal number of clusters for our dataset.")
    # Code to identify the elbow point
    from kneed import KneeLocator
    kn = KneeLocator(k_values, inertia, curve='convex', direction='decreasing')
    elbow_k = kn.knee
    colored_box(f"Feedback: One good candidate for the elbow point in this dataset appears to be at k={elbow_k}. So, the optimal number of clusters for this dataset is likely at or around {elbow_k}. However! The elbow method is not the only tool we should use to select the optimal number of clusters. We can often use this information in conjunction another type of performance evaluator referred to as silhouette scores to make a more informed decision. So, let's do it!", color= "#34d16b")
# Let's also look at the silhouette plot to examine the quality of the clusters.
    from sklearn.metrics import silhouette_score
    silhouette_scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_scaled)
        score = silhouette_score(df_scaled, kmeans.labels_)
        silhouette_scores.append(score)
# Plot the scores, but don't forget to clear the previous plot first to avoid confusion.
    plt.clf()
    plt.plot(range(2, 11), silhouette_scores, marker='o')
    plt.title('Silhouette Scores for K-Means Clustering')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    st.pyplot(plt)
   
    st.write("With silhouette plots, we want to look for the highest silhouette score. A silhouette score closer to 1 tells us that the clusters are properly separated and well-defined. On the other hand, a silhouette score closer to -1 tells us that the clusters may overlap and may not be very well-defined.")
    best_silhouette_k = range(2, 11)[silhouette_scores.index(max(silhouette_scores))]
    colored_box(f"Feedback: The silhouette scores in this graph suggest that the optimal number of clusters for this dataset is {best_silhouette_k}. This is because the silhouette score is highest at k={best_silhouette_k}", color = "#34d16b")
# Allow them to adjust the number of iterations and see how it impacts the results. This is an important hyperparameter to understand
    st.subheader("Other Hyperparameters: Number of Iterations")
    st.write("Now that you have a good understanding of how to select the optimal number of clusters, let's test what the number of iterations does to the results.")
    st.write("The number of iterations is a hyperparameter that determines the maximum number of times the algorithm can run before it must stop. **Increasing the number of iterations can deliver a better solution, but it also increases the time it takes for the algorithm to run.**")
    st.write("K-Means Clustering may stop earlier than this number if it stabilizes first (if the cluster centers do not change significantly between iterations).")
    n_iterations = st.slider("Select the number of iterations for K-Means Clustering", min_value=1, max_value=100, value=10)
    kmeans = KMeans(n_clusters=k, max_iter=n_iterations, random_state=42)
    kmeans.fit(df_scaled)
    st.write(f"K-Means Clustering has been applied with {k} clusters and {n_iterations} iterations. Here are the new cluster centers displayed:")
    st.dataframe(kmeans.cluster_centers_)
    colored_box(f"Feedback: You can see here that adjusting the number of iterations on the lower end of the scale can impact the cluster centers significantly. The larger your dataset, the higher the number of iterations you may need before the cluster centers stabilize. This is an important hyperparameter to keep in mind when using K-Means Clustering.", color="#34d16b")
    st.divider()
# Conclusion, included with a link to learn more about K-means clustering 
    st.subheader("Conclusion")
    st.write("That concludes the K-Means Clustering section! Feel free to keep experimenting with the number of clusters and iterations, or try a new dataset. You can also move on to another technique to explore other methods of unsupervised machine learning.")
    st.write("If you want to learn more about K-Means Clustering, check out this guide from IBM: https://www.ibm.com/think/topics/k-means-clustering")



# END OF K-MEANS CLUSTERING SECTION

# END OF K-MEANS CLUSTERING SECTION

# END OF K-MEANS CLUSTERING SECTION



# Hierarchical Clustering
elif selected_technique == "Hierarchical Clustering":
    st.write("Great! You selected Hierarchical Clustering. This technique builds a hierarchy of clusters by either merging or splitting them based on their similarity.")
    with st.expander("How it works (in more detail):"):
        st.markdown('<div style="background-color: #dac17f; padding: 10px; border-radius: 5px;">How it works (in more detail): Hierarchical clustering can be done in two different ways. The first way is called agglomerative clustering which is a bottom-up method. Under agglomerative clustering, each data point starts as its own cluster, and pairs of clusters are merged as we move up the hierarchy. The second way is called divisive clustering which is a top-down method. With divisive clustering, all data points start in one cluster, and the data points are split into smaller clusters as we move down the hierarchy. We will mainly be looking at agglomerative clustering in this app, but here is a link to learn more about divisive clustering: https://www.geeksforgeeks.org/artificial-intelligence/divisive-clustering/ </div>', unsafe_allow_html=True)
    st.subheader("Linkage Methods and Dendrograms")
    st.write("To get started with hierarchical clustering, please select the linkage method you would like to use. **Linkage methods determine the criterion for measuring the distance between data points when the clusters are actually being merged**. If you're not sure which method to choose first, try Ward's method which minimizes the variance of the clusters that are merged.")
    method_options = ["ward", "complete", "average", "single"]
    selected_method = st.selectbox("Select a method for hierarchical clustering", method_options)
    if selected_method == "ward":
        st.write("You selected Ward's method. This method minimizes the variance of the clusters that are merged, which often leads to more dense clusters.")
    elif selected_method == "complete":
        st.write("You selected Complete Linkage. This method focuses on the maximum distance between points in different clusters. This method also creates more dense clusters.")
    elif selected_method == "average":
        st.write("You selected Average Linkage. This method considers the average distance between points in different clusters. This method can create more balanced clusters.")
    elif selected_method == "single":
        st.write("You selected Single Linkage. This method focuses on the minimum distance between points in different clusters. This method can create more elongated clusters.")
    st.write("Now, please select the number of clusters you would like to display in the dendrogram. A dendrogram is a tree-like diagram that shows the hierarchical relationship between clusters. Dendrograms will help you visualize the hierarchical clustering process. Feel free to keep adjusting the number of clusters and the linkage method to see how it affects the dendrogram.")
    n_clusters = st.slider("Select the number of clusters to display in the dendrogram", min_value=2, max_value=10, value=5)
    # Display the dendrogram to help users visualize the hierarchical clustering process
    from scipy.cluster.hierarchy import dendrogram, linkage
    import matplotlib.pyplot as plt
    linkages = linkage(df_scaled, method=selected_method)
    plt.figure(figsize=(10, 5))
    dendrogram(linkages, truncate_mode='level', p=n_clusters)
    plt.title('Dendrogram for Hierarchical Clustering')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    st.pyplot(plt)
    colored_box("Feedback: You can see that as you increase the number of clusters programmed into the dendrogram, the hierarchy gains more branches and clusters become more specific. You can also see that the different linkage methods create different shapes of clusters. Ward's method and complete linkage create more dense clusters, while average linkage and single linkage create clusters that are more spread out.", color="#34d16b")
    # Silhouette scores can also be calculated for hierarchical clustering to evaluate the quality of the clusters. However, since hierarchical clustering does not assign cluster labels until a certain number of clusters is chosen, we will calculate silhouette scores for a range of cluster numbers and plot them to help users determine the optimal number of clusters.
    st.subheader("Silhouette Scores for Hierarchical Clustering")
    # Make sure to clear the previous plot first to avoid confusion.
    plt.clf()
    silhouette_scores = []
    for k in range(2, 11):
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        from matplotlib import pyplot as plt
        hierarchical = AgglomerativeClustering(n_clusters=k, linkage=selected_method)
        labels = hierarchical.fit_predict(df_scaled)
        score = silhouette_score(df_scaled, labels)
        silhouette_scores.append(score)
    plt.plot(range(2, 11), silhouette_scores, marker='o')
    plt.title('Silhouette Scores for Hierarchical Clustering')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    st.pyplot(plt)
    st.write("Silhouette scores are a measure of how similar a data point is to its own cluster compared to other clusters. They are a helpful measure for evaluating the quality of the clusters.")
    st.write("With silhouette plots, we want to look for the highest silhouette score. A silhouette score closer to 1 tells us that the clusters are properly separated and well-defined. On the other hand, a silhouette score closer to -1 tells us that the clusters may overlap and may not be very well-defined.")
    best_k = range(2, 11)[silhouette_scores.index(max(silhouette_scores))]
    colored_box(f"Feedback: The silhouette scores here suggest that the optimal number of clusters for this dataset is {best_k}. This is because the silhouette score is highest at this number of clusters.", color="#34d16b")
    
    # Allow the user to experiment with hyperparameters of hierarchical clustering
    st.subheader("Other Hyperparameters: Truncation Modes for Dendrograms")
    
    from scipy.cluster.hierarchy import dendrogram, linkage
    import matplotlib.pyplot as plt
    linked = linkage(df_scaled, method=selected_method)
    # Clear plot
    plt.clf()
    
    # Truncation hyperparameter code
    st.write("Sometimes it becomes necessary to truncate the dendrogram to be able to visualize it effectively (especially if the dataset you've chosen is large). Try adjusting the truncate mode to see how it impacts the dendrogram.")

    st.write("Here is a brief guide to the truncation modes:")
    st.write("1. 'Lastp' truncation will display only the last p merged clusters.")
    st.write("2. 'Level' truncation will display clusters up to a specified level in the hierarchy.")
    st.write("3. 'None' will display the full dendrogram without any truncation.")

    truncate_mode = st.selectbox("Select truncate mode", [ "lastp", "none", "level"])
    if truncate_mode == "lastp":
        p = st.slider("Select the number of clusters to display", min_value=2, max_value=10, value=5)
        st.write(f"You selected 'lastp' truncation with p={p}. This means that the dendrogram will display only the last {p} merged clusters, which can help you focus on the most recent merges in the hierarchy.")
    elif truncate_mode == "level":
        p = st.slider("Select the level to truncate at", min_value=0, max_value=10, value=5)
        st.write(f"You selected 'level' truncation with p={p}. This means that the dendrogram will display clusters up to level {p} in the hierarchy, which can help you focus on the broader structure of the clusters.")
    else:
        p = 0  # p is not used when truncate_mode is 'none'
        st.write("You selected 'none' truncation. This means that the full dendrogram will be displayed without any truncation, which can help you see the complete hierarchy of clusters.")
    plt.figure(figsize=(10, 7))
    dendrogram(linked, truncate_mode=truncate_mode, p=p)
    plt.title(f'Truncated Hierarchical Clustering Dendrogram ({selected_method} method)')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    st.pyplot(plt)
    colored_box("As you can see, adjusting the truncate mode and p parameters allows you to focus on different levels of the hierarchy in the dendrogram. This can help you better understand the structure of the clusters and how they are formed at different levels of the hierarchy.", color="#34d16b")
    st.divider()

# Conclusion, including a link to learn more about hierarchical clustering
    st.subheader("Conclusion")
    st.write("That concludes the Hierarchical Clustering section! Feel free to keep experimenting with the linkage methods, number of clusters, and truncation modes, or try a new dataset. You can also move on to another technique to explore other methods of unsupervised machine learning.")
    st.write("If you want to learn more about Hierarchical Clustering, check out this guide I find really helpful from DataCamp: https://www.datacamp.com/tutorial/hierarchical-clustering")



# END OF HIERARCHICAL CLUSTERING SECTION

# END OF HIERARCHICAL CLUSTERING SECTION

# END OF HIERARCHICAL CLUSTERING SECTION



# Principle Componenent Analysis (PCA)
elif selected_technique == "Principal Component Analysis (PCA)":
    from sklearn.decomposition import PCA
    st.write("Great! You selected Principal Component Analysis (PCA). This technique is used for dimensionality reduction, which helps to simplify the dataset while retaining as much variance as possible.")
    with st.expander("How it works (in more detail):"):
        st.markdown('<div style="background-color: #dac17f; padding: 10px; border-radius: 5px;">PCA works by identifying the directions (called principal components) in which the data varies the most. The first principal component captures the most variance in the data, the second principal component captures the second most variance, and so on. By selecting a subset of these principal components, we can reduce the dimensionality of the dataset while retaining as much of the original variance as possible.</div>', unsafe_allow_html=True)
    st.write("To get started with PCA, please select the number of principal components you would like to compute. The number of principal components determines how many new features will be created from the original features in the dataset.")
    max_components = min(df_scaled.shape[0], df_scaled.shape[1])
    n_components = st.slider("Select the number of principal components", 1, max_components, 2)
    if df_scaled.shape[0] < n_components:
        st.write(f"The number of principal components cannot exceed the number of original features. Please select a number of principal components less than or equal to {df_encoded.shape[0]}.")
    else:
        import matplotlib.pyplot as plt
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(df_scaled)
        st.write(f"PCA has been applied with {n_components} principal components. Here are the explained variance ratios for each principal component:")
        st.dataframe(pca.explained_variance_ratio_)
        st.write("The explained variance ratio shows the proportion of the dataset's variance that is captured by each principal component. A higher explained variance ratio is better because it means that the principal component captures more of the variance in the dataset.")
        #Visualize the principal components with a scatter plot if n_components is 2 or 3.
        if n_components == 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
            plt.title('PCA Scatter Plot (2 Principal Components)')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            st.pyplot(plt)
        elif n_components == 3:
            # In case the user selects 3 components, import 3D plot tools from matplotlib's 3D toolkit
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(principal_components[:, 0], principal_components[:, 1], principal_components[:, 2], alpha=0.5)
            ax.set_title('PCA Scatter Plot (3 Principal Components)')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_zlabel('Principal Component 3')
            st.pyplot(plt)

        colored_box("Feedback: The scatter plot of the principal components can help you visualize the structure of the data in a lower-dimensional space. If you see distinct clusters or patterns in the scatter plot, it may indicate that the PCA has successfully captured important variance in the dataset.", color="#34d16b")

        # Look at loadings to understand which original features contribute the most to each principal component.
        st.subheader("Loadings for Principal Components")
        loadings = pd.DataFrame(pca.components_.T, columns=[f'Principal Component {i+1}' for i in range(n_components)], index=df_encoded.columns)
        st.write("Here are the loadings for each original feature on the principal components:")    
        st.dataframe(loadings)
        st.write("The loadings show how much each original feature contributes to each principal component. A higher absolute value of a loading indicates that the original feature has a stronger influence on the corresponding principal component.")
        colored_box("Feedback: By examining the loadings, you can gain insights into which original features are most important in explaining the variance captured by each principal component. This can help you understand the underlying structure of the data and identify which features are driving the patterns observed in the PCA scatter plot.", color="#34d16b")
        # Clear the plot
        plt.clf()
        st.subheader("Visualizing Loadings with a Horizontal Bar Chart")
        st.write("We can also visualize the loadings with a horizontal grouped bar chart to make it easier to compare the contributions of the original features across the principal components.")
        loadings.plot(kind='barh', figsize=(10, 8))
        st.pyplot(plt)
        colored_box("Feedback: With the horizontal bar chart, you can more easily see which features have the highest loadings for each principal component and how they compare to each other.", color="#34d16b")
        st.divider()
        # Conclusion. including a link to learn more about principal component analysis
        st.subheader("Conclusion")
        st.write("That concludes the Principal Component Analysis (PCA) section! Feel free to keep experimenting with the number of principal components, or try a new dataset. You can also move on to another technique to explore other methods of unsupervised machine learning.")
        st.write("If you want to learn more about PCA, check out this guide from the AI company Turing: https://www.turing.com/kb/guide-to-principal-component-analysis")

    

# END OF PCA SECTION

# END OF PCA SECTION

# END OF PCA SECTION



# End of App