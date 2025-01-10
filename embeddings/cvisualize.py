import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
neighbor_folder_path = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, neighbor_folder_path)

from whisper_live.cluster_embeddings import SpeakerEmbeddingClassifierWithClustering

import time


class AudioEmbeddingVisualizerWithHover:
    def __init__(self, embeddings, speaker_ids, file_labels, file_names):
        """
        Initialize the visualizer with embeddings, speaker IDs, file labels, and file names.

        Parameters:
        - embeddings: numpy array of shape (n_samples, n_features)
        - speaker_ids: list of speaker IDs for each embedding
        - file_labels: list of integers indicating the source file for each embedding
        - file_names: list of file names corresponding to each label
        """
        self.embeddings = embeddings
        self.speaker_ids = speaker_ids
        self.file_labels = file_labels
        self.file_names = file_names

    def visualize_embeddings(self):
        """
        Visualize embeddings using t-SNE in 2D, with points colored by speaker ID.
        """
        # Determine a valid perplexity
        num_samples = self.embeddings.shape[0]
        perplexity = min(30, num_samples - 1)  # Perplexity must be less than the number of samples

        # Perform t-SNE to reduce dimensions to 2D
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        reduced_embeddings = tsne.fit_transform(self.embeddings)

        # Generate a distinct color for each speaker
        unique_speaker_ids = sorted(set(self.speaker_ids))
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(unique_speaker_ids))]
        speaker_to_color = {speaker_id: colors[i] for i, speaker_id in enumerate(unique_speaker_ids)}

        # Assign colors based on speaker IDs
        point_colors = [speaker_to_color[speaker_id] for speaker_id in self.speaker_ids]

        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=point_colors, s=50, alpha=0.7)

        ax.set_title("t-SNE Visualization of Embeddings Colored by Speaker ID")
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")

        # Add a legend with speaker IDs
        for speaker_id, color in speaker_to_color.items():
            ax.scatter([], [], c=[color], label=f"Speaker {speaker_id}")
        ax.legend(title="Speakers", loc="upper right")

        # Add mouse-over feature
        self.add_hover_tooltip(fig, ax, reduced_embeddings, scatter)

        plt.show()

    def add_hover_tooltip(self, fig, ax, reduced_embeddings, scatter):
        """
        Add a mouse-over feature to display the original filename for each point.

        Parameters:
        - fig: matplotlib figure object
        - ax: matplotlib axis object
        - reduced_embeddings: 2D t-SNE reduced embeddings
        """
        annot = ax.annotate("", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        def update_annot(ind):
            """Update annotation with the filename of the hovered point."""
            idx = ind["ind"][0]
            filename = self.file_names[self.file_labels[idx]]
            pos = reduced_embeddings[idx]
            annot.xy = pos
            annot.set_text(f"File: {filename}")
            annot.get_bbox_patch().set_facecolor("yellow")
            annot.get_bbox_patch().set_alpha(0.8)

        def on_hover(event):
            """Event handler for mouse movement."""
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = scatter.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                elif vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_hover)


def load_embeddings_with_clustering(file_paths):
    """
    Load embeddings from multiple files, classify them using the clustering classifier,
    and associate each embedding with its source file and speaker ID.

    Parameters:
    - file_paths: list of strings, paths to embedding files.

    Returns:
    - numpy array of combined embeddings.
    - list of speaker IDs.
    - list of file labels for each embedding.
    - list of file names.
    """
    classifier = SpeakerEmbeddingClassifierWithClustering(similarity_threshold=0.6, clustering_eps=0.4)
    all_embeddings = []
    file_labels = []
    file_names = [file_path for file_path in file_paths]  # Preserve file names

    # Load embeddings from files
    for file_idx, file_path in enumerate(file_paths):
        with open(file_path, "r") as f:
            for line in f:
                embedding = list(map(float, line.strip().split()))
                embedding = np.array(embedding)
                all_embeddings.append(embedding)
                file_labels.append(file_idx)  # Assign a unique label for each file
    start_time = time.time()

    classifier.bulk_add_embeddings(all_embeddings)
    speaker_ids = classifier.get_classifications()
    all_embeddings = np.array(all_embeddings)
    print(f"Time taken to cluster all {len(all_embeddings)} embeddings is {time.time() - start_time}")
    return all_embeddings, speaker_ids, file_labels, file_names


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Visualize audio embeddings with t-SNE and hover tooltips.")
    parser.add_argument("files", nargs="+", help="Paths to one or more embedding files")
    args = parser.parse_args()

    # Load embeddings from the specified files
    all_embeddings, speaker_ids, file_labels, file_names = load_embeddings_with_clustering(args.files)

    # Initialize and visualize embeddings
    visualizer = AudioEmbeddingVisualizerWithHover(all_embeddings, speaker_ids, file_labels, file_names)
    visualizer.visualize_embeddings()
