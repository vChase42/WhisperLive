import numpy as np

class SpeakerClustering:
    def __init__(self, similarity_threshold=0.7):
        self.clusters = []  # Each cluster is a dictionary with 'centroid' and 'embeddings'
        self.similarity_threshold = similarity_threshold

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def add_embedding(self, embedding):
        if not isinstance(embedding, np.ndarray):
            raise ValueError("Embedding must be a numpy array.")

        if len(self.clusters) == 0:
            new_cluster = {
                'centroid': embedding.copy(),
                'embeddings': [embedding.copy()]
            }
            self.clusters.append(new_cluster)
            return

        similarities = []
        for cluster in self.clusters:
            centroid = cluster['centroid']
            sim = self._cosine_similarity(embedding, centroid)
            similarities.append((sim, cluster))

        # Sort clusters by similarity in descending order
        similarities.sort(reverse=True, key=lambda x: x[0])

        target_cluster = None
        for sim, cluster in similarities:
            if sim >= self.similarity_threshold:
                target_cluster = cluster
                break

        if target_cluster is not None:
            target_cluster['embeddings'].append(embedding.copy())
            # Update centroid by taking the average of all embeddings in the cluster
            target_cluster['centroid'] = np.mean(target_cluster['embeddings'], axis=0)
        else:
            new_cluster = {
                'centroid': embedding.copy(),
                'embeddings': [embedding.copy()]
            }
            self.clusters.append(new_cluster)

        # Check for possible cluster merging after adding the new embedding
        self._merge_similar_clusters()

    def _merge_similar_clusters(self):
        num_clusters = len(self.clusters)
        if num_clusters <= 1:
            return

        # Compute pairwise similarities between centroids
        similarity_matrix = np.zeros((num_clusters, num_clusters))
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                centroid_i = self.clusters[i]['centroid']
                centroid_j = self.clusters[j]['centroid']
                sim = self._cosine_similarity(centroid_i, centroid_j)
                similarity_matrix[i][j] = sim
                similarity_matrix[j][i] = sim

        # Find pairs of clusters to merge
        to_merge = []
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                if similarity_matrix[i][j] >= self.similarity_threshold:
                    to_merge.append((i, j))

        # Merge clusters in reverse order to prevent index issues
        merged_indices = set()
        for pair in reversed(to_merge):
            i, j = pair
            if i not in merged_indices and j not in merged_indices:
                # Merge cluster j into cluster i
                self.clusters[i]['embeddings'].extend(self.clusters[j]['embeddings'])
                self.clusters[i]['centroid'] = np.mean(
                    self.clusters[i]['embeddings'], axis=0
                )
                del self.clusters[j]
                merged_indices.add(i)
                merged_indices.add(j)

    def get_clusters(self):
        return [cluster['embeddings'] for cluster in self.clusters]

    def clear(self):
        self.clusters = []