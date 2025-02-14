from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SpeakerClustering:
    def __init__(self, similarity_threshold=0.75, clustering_eps=0.5, clustering_min_samples=2):
        """
        Initializes the SpeakerEmbeddingClassifier with clustering.
        Args:
            similarity_threshold (float): Threshold for direct cosine similarity matching.
            clustering_eps (float): Maximum distance between two samples for DBSCAN clustering.
            clustering_min_samples (int): Minimum samples required to form a cluster in DBSCAN.
        """
        self.similarity_threshold = similarity_threshold
        self.clustering_eps = clustering_eps
        self.clustering_min_samples = clustering_min_samples

        self.embeddings = []  # List of all embeddings in the order they were added
        self.original_order = []  # Tracks the order of embeddings for classification
        self.speakers = {}  # Speaker ID -> list of embeddings
        self.next_speaker_id = 0  # Counter for new speaker IDs
        self.classifications = []  # Speaker classifications in original order

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
        Reclusters all embeddings using DBSCAN and updates speaker IDs.
        """
        if len(self.embeddings) < self.clustering_min_samples:
            return  # Not enough embeddings for clustering

        # Perform clustering with DBSCAN
        clustering = DBSCAN(
            metric="cosine", eps=self.clustering_eps, min_samples=self.clustering_min_samples
        ).fit(self.embeddings)

        labels = clustering.labels_

        # Reset speakers and classifications
        self.speakers = {}
        self.classifications = []

        # Assign embeddings to speakers based on DBSCAN labels
        for idx, label in enumerate(labels):
            if label == -1:
                # Noise embeddings (not part of any cluster)
                self.classifications.append(-1)
                continue
            if label not in self.speakers:
                self.speakers[label] = []
            self.speakers[label].append(self.embeddings[idx])
            self.classifications.append(label)

        # Reassign consistent speaker IDs and update next speaker ID
        new_speakers = {}
        for new_id, (old_id, embeddings) in enumerate(self.speakers.items()):
            new_speakers[new_id] = embeddings
            self.classifications = [new_id if c == old_id else c for c in self.classifications]
        self.speakers = new_speakers
        self.next_speaker_id = len(self.speakers)

    def add_and_classify_embedding(self, embedding):
        """
        Adds a new embedding and classifies it based on similarity or clustering.
        Args:
            embedding (np.ndarray): The new embedding to classify and add.
        Returns:
            int: Speaker ID of the embedding (-1 if it is noise or not part of any cluster).
        """
        # Attempt to match with existing speakers using similarity threshold
        for speaker_id, embeddings in self.speakers.items():
            similarity = self._calculate_similarity(embedding, embeddings)
            if similarity >= self.similarity_threshold:
                self.speakers[speaker_id].append(embedding)
                self.embeddings.append(embedding)
                self.original_order.append(len(self.embeddings) - 1)
                self.classifications.append(speaker_id)
                return speaker_id

        # Add embedding and recluster
        self.embeddings.append(embedding)
        self.original_order.append(len(self.embeddings) - 1)
        self._recluster_embeddings()

        # Get the speaker ID after reclustering
        print("bruh:",len(self.classifications))
        if(len(self.classifications) == 0): return -1
        return self.classifications[-1]

    def bulk_add_embeddings(self, embeddings):
        """
        Adds multiple embeddings and reclassifies all points.
        Args:
            embeddings (list of np.ndarray): List of embeddings to add.
        """
        self.embeddings.extend(embeddings)
        self.original_order.extend(range(len(self.embeddings) - len(embeddings), len(self.embeddings)))
        self._recluster_embeddings()

    def get_classifications(self):
        """
        Returns the speaker classifications for all embeddings in the original order.
        Returns:
            list of int: List of speaker IDs in the order the embeddings were added.
        """
        return [self.classifications[idx] for idx in self.original_order]

    def get_all_embeddings(self):
        """
        Retrieves all stored embeddings.
        Returns:
            list of np.ndarray: List of all embeddings.
        """
        return self.embeddings

    def get_all_speaker_ids(self):
        """
        Retrieves all stored speaker IDs.
        Returns:
            list: A list of all speaker IDs.
        """
        return list(self.speakers.keys())