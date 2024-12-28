from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SpeakerEmbeddingClassifierWithClustering:
    def __init__(self, similarity_threshold=0.75, clustering_eps=0.5, clustering_min_samples=2):
        """
        Initializes the SpeakerEmbeddingClassifier with clustering.
        
        Args:
            similarity_threshold (float): Threshold for direct cosine similarity matching.
            clustering_eps (float): Maximum distance between two samples for DBSCAN clustering.
            clustering_min_samples (int): Minimum samples required to form a cluster in DBSCAN.
        """
        self.speakers = {}  # Dictionary to store speaker IDs and their embeddings
        self.next_speaker_id = 0  # Counter for assigning new speaker IDs
        self.similarity_threshold = similarity_threshold  # Cosine similarity threshold
        self.embeddings = []  # Flat list of all embeddings for clustering
        self.clustering_eps = clustering_eps  # DBSCAN epsilon parameter
        self.clustering_min_samples = clustering_min_samples  # DBSCAN min_samples parameter

    def _calculate_similarity(self, embedding, embeddings):
        """
        Calculates cosine similarity between the new embedding and a set of embeddings.
        
        Args:
            embedding (np.ndarray): The new embedding to compare.
            embeddings (list of np.ndarray): The embeddings to compare against.
        
        Returns:
            float: Maximum similarity score with the provided embeddings.
        """
        if len(embeddings) == 0:
            return 0  # No similarity if no embeddings exist
        return np.max(cosine_similarity([embedding], embeddings)[0])

    def _recluster_embeddings(self):
        """
        Reclusters all embeddings to assign consistent speaker IDs.
        """
        if len(self.embeddings) < self.clustering_min_samples:
            return  # Not enough embeddings for clustering

        # Perform clustering using DBSCAN
        clustering = DBSCAN(
            metric="cosine", eps=self.clustering_eps, min_samples=self.clustering_min_samples
        ).fit(self.embeddings)

        # Update speakers based on clustering labels
        self.speakers = {}
        labels = clustering.labels_

        for idx, label in enumerate(labels):
            if label == -1:
                # Noise embeddings (no cluster)
                continue
            if label not in self.speakers:
                self.speakers[label] = []
            self.speakers[label].append(self.embeddings[idx])

        # Reassign consistent speaker IDs
        self.speakers = {
            speaker_id: embeddings
            for speaker_id, (speaker_id, embeddings) in enumerate(self.speakers.items())
        }
        self.next_speaker_id = len(self.speakers)

    def add_and_classify_embedding(self, embedding):
        """
        Adds a new embedding and classifies it based on similarity or clustering.
        
        Args:
            embedding (np.ndarray): The new embedding to classify and add.
        
        Returns:
            int: -1 if a new speaker is added, or the existing speaker ID.
        """
        # Try direct similarity matching first
        for speaker_id, embeddings in self.speakers.items():
            similarity = self._calculate_similarity(embedding, embeddings)
            if similarity >= self.similarity_threshold:
                self.speakers[speaker_id].append(embedding)
                self.embeddings.append(embedding)
                return speaker_id

        # Add to global embeddings and recluster
        self.embeddings.append(embedding)
        self._recluster_embeddings()

        # Check if embedding was assigned to an existing cluster
        for speaker_id, embeddings in self.speakers.items():
            if any(np.array_equal(embedding, e) for e in embeddings):
                return speaker_id

        # If not, assign a new speaker ID
        new_speaker_id = self.next_speaker_id
        self.speakers[new_speaker_id] = [embedding]
        self.next_speaker_id += 1
        return new_speaker_id

    def get_all_embeddings(self):
        """
        Retrieves all stored embeddings.
        
        Returns:
            dict: A dictionary of speaker IDs and their embeddings.
        """
        return self.speakers

    def get_all_speaker_ids(self):
        """
        Retrieves all stored speaker IDs.
        
        Returns:
            list: A list of all stored speaker IDs.
        """
        return list(self.speakers.keys())
