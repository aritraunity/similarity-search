import math
import numpy as np
import scipy
import torch
from sentence_transformers import SentenceTransformer
from similaritysearch.math import (euclidean_distance, 
                                   scipy_euclidean_distance, 
                                   cosine_similarity, 
                                   matrix_cosine_similarity,
                                   matrix_cosine_similarity_multi)
import matplotlib.pyplot as plt

documents = [
    'Bugs introduced by the intern had to be squashed by the lead developer.',
    'Bugs found by the quality assurance engineer were difficult to debug.',
    'Bugs are common throughout the warm summer months, according to the entomologist.',
    'Bugs, in particular spiders, are extensively studied by arachnologists.'
]
#Cache Path to Download sentence transformer model
#paraphrase-MiniLM-L6-v2: Trained to generate similar vectors for sentences with similar semantic meaning
cache_path = '/home/aritra-mukherjee/projects/similarity-search/cache/'
model = SentenceTransformer('paraphrase-MiniLM-L6-v2', cache_folder=cache_path)
embeddings = model.encode(documents)
print(f"Shape: {embeddings.shape}")
#The total number of distances between 4 vectors, applying P&C: 4 x 4 = 16 values
#Define a 4 x 4 vectors with all zeroes
x_dim = embeddings.shape[0]
l2_distances = np.zeros([x_dim, x_dim])
#Calculating the distance between all Vectors in embeddings
# Total Number of Vectors = 4
for i in range(x_dim):
    for j in range(x_dim):
        l2_distances[i, j] = euclidean_distance(embeddings[i], embeddings[j])

print("Slower / Brute forced calculation of L2 Distances")
print(l2_distances)
#Improvising
l2_distances_improved = np.zeros([4,4])
for i in range(x_dim):
    for j in range(x_dim):
        """
        Distance matrices are symmetrical,
        You only need to calculate the upper row because euclidean distances are symmetric.
        (SKIP) (0, 1) (0, 2) (0, 3)
        (SKIP) (SKIP) (1, 2) (1, 3)
        (SKIP) (SKIP) (SKIP) (2, 3)
        We only calculate ~30% of the values for a 4 x 4 matrix
        """
        if j > i:
            l2_distances_improved [i, j]= euclidean_distance(embeddings[i], embeddings[j])
        #For the lower triangle, we copy the already calculated values to make it perfectly symmetric
        #Euclidean distances are symmetric D(a, b) = D(b, a)
        elif i > j:
            l2_distances_improved [i, j] = l2_distances_improved[j, i]

print("Improved Calculations on L2 Distance")
print(l2_distances_improved)
print("L2 / Euclidean Distance using Sci Py: Analytical comparision")
l2_distance_scipy = scipy_euclidean_distance(embeddings=embeddings)
print(l2_distance_scipy)
print(np.allclose(l2_distance_scipy, l2_distances))

# Statement semantic similarity with a document
print("Check Cosine Similarity")
source = ["Who is responsible for a coding project and fixing others' mistakes?"]
source_embedding = model.encode(source)
print(f"Source Shape: {source_embedding.shape} \n Embeddings Shape: {embeddings.shape}")
cosine_similarities = []
for i in range(x_dim):
    x = cosine_similarity(source_embedding[0], embeddings[i])
    cosine_similarities.append(x)

max_cossm = max(cosine_similarities)
max_cossm_id = cosine_similarities.index(max_cossm)
print (f"Q: {source[0]}\nA:{documents[max_cossm_id]}")
print("Calculation of Co-Sine Similarities using PyTorch for Normalization and Matrix Dot Product")
cs = matrix_cosine_similarity(embeddings=embeddings)
print(cs)
print("Co-Sine Similarities for Source vs. Document embeddings")
cs_1 = matrix_cosine_similarity_multi(embeddings1=embeddings, embeddings2=source_embedding)
print(cs_1)

# Data Visualization
doc_labels = [f"Doc {i + 1}" for i in range(len(documents))]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Similarity Search — Visual Analysis")

# L2 Distance Matrix heatmap
im0 = axes[0, 0].imshow(l2_distances, cmap='YlOrRd')
axes[0, 0].set_title("L2 Distance Matrix")
axes[0, 0].set_xticks(range(len(doc_labels)))
axes[0, 0].set_yticks(range(len(doc_labels)))
axes[0, 0].set_xticklabels(doc_labels, rotation=45, ha='right')
axes[0, 0].set_yticklabels(doc_labels)
fig.colorbar(im0, ax=axes[0, 0])
for i in range(len(doc_labels)):
    for j in range(len(doc_labels)):
        axes[0, 0].text(j, i, f"{l2_distances[i, j]:.2f}", ha='center', va='center', fontsize=8)

# Matrix Cosine Similarity heatmap (documents vs documents)
im1 = axes[0, 1].imshow(cs, cmap='coolwarm', vmin=-1, vmax=1)
axes[0, 1].set_title("Matrix Cosine Similarity (Documents)")
axes[0, 1].set_xticks(range(len(doc_labels)))
axes[0, 1].set_yticks(range(len(doc_labels)))
axes[0, 1].set_xticklabels(doc_labels, rotation=45, ha='right')
axes[0, 1].set_yticklabels(doc_labels)
fig.colorbar(im1, ax=axes[0, 1])
for i in range(len(doc_labels)):
    for j in range(len(doc_labels)):
        axes[0, 1].text(j, i, f"{cs[i, j]:.2f}", ha='center', va='center', fontsize=8)

# Cosine Similarity — query vs documents (manual loop)
axes[1, 0].bar(doc_labels, cosine_similarities, color='steelblue')
axes[1, 0].set_title("Query vs. Document Cosine Similarity (Manual)")
axes[1, 0].set_xticks(range(len(doc_labels)))
axes[1, 0].set_xticklabels(doc_labels, rotation=45, ha='right')
axes[1, 0].set_ylabel("Cosine Similarity")
axes[1, 0].set_ylim(0, max(cosine_similarities) * 1.2)

# Cosine Similarity — query vs documents (matrix)
axes[1, 1].bar(doc_labels, cs_1.flatten(), color='coral')
axes[1, 1].set_title("Query vs. Document Cosine Similarity (Matrix)")
axes[1, 1].set_xticks(range(len(doc_labels)))
axes[1, 1].set_xticklabels(doc_labels, rotation=45, ha='right')
axes[1, 1].set_ylabel("Cosine Similarity")
axes[1, 1].set_ylim(0, max(cs_1.flatten()) * 1.2)

plt.tight_layout()
plt.show()