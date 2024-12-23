import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import argparse

class AudioEmbeddingVisualizer:
    def __init__(self, embeddings, file_labels, file_names):
        """
        Initialize the visualizer with embeddings, their corresponding file labels, and file names.

        Parameters:
        - embeddings: numpy array of shape (n_samples, n_features)
        - file_labels: list of integers indicating the source file for each embedding
        - file_names: list of file names corresponding to each label
        """
        self.embeddings = embeddings
        self.file_labels = file_labels
        self.file_names = file_names

    def visualize_embeddings(self):
        """
        Visualize embeddings using t-SNE in 2D, with points colored by their source file.
        """
        # Determine a valid perplexity
        num_samples = self.embeddings.shape[0]
        perplexity = min(30, num_samples - 1)  # Perplexity must be less than the number of samples

        # Perform t-SNE to reduce dimensions to 2D
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        reduced_embeddings = tsne.fit_transform(self.embeddings)

        # Generate a distinct color for each file
        unique_labels = sorted(set(self.file_labels))
        cmap = plt.get_cmap("tab10")  # Use a colormap with distinct colors
        colors = [cmap(i % 10) for i in range(len(unique_labels))]
        label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}

        # Assign colors based on file labels
        point_colors = [label_to_color[label] for label in self.file_labels]

        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=point_colors, s=50, alpha=0.7)

        ax.set_title("t-SNE Visualization of Embeddings Colored by File")
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")

        # Add a legend with file names
        for label, color in label_to_color.items():
            ax.scatter([], [], c=[color], label=self.file_names[label])
        ax.legend(title="Source Files", loc="upper right")

        plt.show()


def load_embeddings_from_files(file_paths):
    """
    Load embeddings from multiple files and associate each embedding with its source file.

    Parameters:
    - file_paths: list of strings, paths to embedding files.

    Returns:
    - numpy array of combined embeddings.
    - list of file labels for each embedding.
    - list of file names.
    """
    all_embeddings = []
    file_labels = []
    file_names = [file_path for file_path in file_paths]  # Preserve file names
    for file_idx, file_path in enumerate(file_paths):
        with open(file_path, "r") as f:
            for line in f:
                embedding = list(map(float, line.strip().split()))
                all_embeddings.append(embedding)
                file_labels.append(file_idx)  # Assign a unique label for each file
    return np.array(all_embeddings), file_labels, file_names


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Visualize audio embeddings with t-SNE.")
    parser.add_argument("files", nargs="+", help="Paths to one or more embedding files")
    args = parser.parse_args()

    # Load embeddings from the specified files
    all_embeddings, file_labels, file_names = load_embeddings_from_files(args.files)

    # Initialize and visualize embeddings
    visualizer = AudioEmbeddingVisualizer(all_embeddings, file_labels, file_names)
    visualizer.visualize_embeddings()
