import numpy as np
from pyannote.core import Segment, SlidingWindowFeature
from pyannote.audio.pipelines.clustering import AgglomerativeClustering
from pyannote.core import SlidingWindow  # ensure imported


#requirements
#add an embedding and then retrieve classification for that embedding

#when reclustering, if old embeddings get reclassified, should they be updated in the logs?
#when reclustering, if two 'distinct' clusters become close (e.g, they were 1 speaker all along), handle a merge?


class SpeakerClustering:
    def __init__(self, min_cluster_size = 2, threshold=0.7, method="weighted"):
        #    metric : {"cosine", "euclidean", ...}, optional
        #    method : {"average", "centroid", "complete", "median", "single", "ward"}
        self.clusterer = AgglomerativeClustering(metric="cosine", max_num_embeddings=np.inf)
        self.clusterer.min_cluster_size = min_cluster_size   
        self.clusterer.threshold = threshold         
        self.clusterer.method = method    


    def cluster_embeddings(self,embeddings):
        embeddings_array = np.array(embeddings)
        embeddings_array = np.atleast_2d(embeddings_array)  # ensure shape (N, D)
        embeddings_array = embeddings_array.reshape(embeddings_array.shape[0], 1, embeddings_array.shape[1])
        
        # --- Create dummy segmentation ---
        # Since we assume every extracted embedding is valid, we mark them as active.
        N = embeddings_array.shape[0]
        dummy_data = np.ones((N, 1, 1))
        dummy_window = SlidingWindow(start=0, duration=1, step=1)
        dummy_segmentation = SlidingWindowFeature(data=dummy_data, sliding_window=dummy_window)
        
        # --- Clustering ---

        hard_clusters, soft_clusters, centroids = self.clusterer(embeddings_array, segmentations=dummy_segmentation)

        return hard_clusters, soft_clusters
