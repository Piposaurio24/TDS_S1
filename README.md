# Spotify Song Recommender with Clustering

A song recommendation system built with unsupervised learning, for **Solemne 1** of the TDS course (Universidad San Sebastián). The notebook uses clustering to fill in genre labels for a large song library, then recommends 10 new songs to each of 6 users based on the genres of the songs they already like.

**Notebook:** `Solemne_1_TDS.ipynb`

## Problem

Spotify users say the platform's recommendations don't match the songs they have voted for or listened to. As a newly hired data scientist, the task is to build a recommender based on each person's tastes, using clustering or association techniques.

## Data

| File | Description |
|---|---|
| `spotify_dataset.csv` | 42,305 Spotify songs described by 23 features (danceability, energy, loudness, acousticness, etc.) |
| `labeled_songs.csv` | 150 songs from the dataset with the same features plus a genre label |
| `user_preferences.csv` | For each of 6 users (indices 0 to 5), the indices of the songs from `spotify_dataset.csv` that they liked |

The data files are not included in this repository. Upload them to Colab (the notebook reads them from `/content/`) or adjust the paths in the notebook.

## Approach

1. **Load and prepare the data**
   - Load the three CSV files with pandas.
   - Merge the 150 labeled songs into the main dataset by `song_index` to attach genres (only 150 of the 42,305 songs have a genre).
   - Rename the `user_preferences` columns to `user_id` and `song_index`.
   - Standardize the numerical features with `StandardScaler`: danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms and time_signature.
   - Drop identifier and text columns (`id`, `uri`, `track_href`, `analysis_url`, `title`, `song_name`, `type` and index columns) before clustering.

2. **Cluster the songs with two algorithms**
   - **K-Means**
   - **Agglomerative Clustering** (Ward linkage)

3. **Choose the number of clusters**
   - K-Means: the Elbow Method (inertia) and the Silhouette score, comparing a range of values of k.
   - Agglomerative Clustering: the Silhouette score.

4. **Assign genres to every song**
   For each cluster, the most frequent genre among the labeled songs that fall in it is assigned to all songs in that cluster. A cluster with no labeled songs is marked `unknown`.

5. **Recommend 10 songs per user** (done once per algorithm)
   - Find the genres of the songs the user liked, ignoring `unknown`.
   - Remove every song the user already liked from the candidates.
   - Pick 10 songs at random (`random_state=42`) from the remaining songs whose genre matches one of the user's genres.
   - If fewer than 10 matching songs are available, or the user has no known genres, fall back to a random sample of unlistened songs.

6. **Compare the two algorithms** and discuss why the recommendations differ or not.

7. **Change the method for choosing the number of clusters** and re-run the clustering and genre assignment, to see how the choice of k changes the clusters, genres and recommendations.

## Number of clusters

| Algorithm | First choice | After changing the method |
|---|---|---|
| K-Means | k = 5 (Elbow Method, with k = 6 also reasonable) | k = 7 (Silhouette plot) |
| Agglomerative Clustering | k = 4 | k = 2 |

For K-Means the silhouette scores were generally low with no sharp peak, which suggests the songs don't form very well-separated groups in this feature space.

## Tech stack

- Python 3
- pandas
- scikit-learn (`StandardScaler`, `KMeans`, `AgglomerativeClustering`, `silhouette_score`)
- matplotlib

## How to run

1. Open `Solemne_1_TDS.ipynb` in Google Colab (or Jupyter).
2. Upload `spotify_dataset.csv`, `labeled_songs.csv` and `user_preferences.csv`, and update the file paths in the notebook if yours have different names.
3. Run the cells in order. Agglomerative Clustering and the silhouette scores on 42,305 songs can take a while.

## Limitations

- Only 150 songs are labeled, so the genre given to each cluster rests on a small sample, and with few clusters each "genre" covers a very broad range of songs.
- Recommendations are chosen randomly among songs with a matching genre, so they depend on the random seed and are not ranked by similarity to what the user liked.
- Using audio features alone, the clusters are only loosely related to musical genre.
- Some cells in the notebook were re-run several times to fix scope errors, and the saved outputs for the final recommendation lists are not all present. Run the notebook again to regenerate them.

## Possible improvements

- Rank candidates by distance to the user's liked songs instead of picking randomly within a genre.
- Try other methods for choosing the number of clusters (for example the Davies-Bouldin or Calinski-Harabasz indices) or a dendrogram for the hierarchical model.
- Use the labeled songs more directly, for example with a supervised classifier or semi-supervised learning.
