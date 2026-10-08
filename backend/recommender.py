import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD


class HybridRecommender:

    def __init__(self):
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        data_dir = os.path.join(base, "ml-1m")

        self.ratings = pd.read_csv(
            os.path.join(data_dir, "ratings.dat"),
            sep="::",
            engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"],
            encoding="latin-1"
        )

        self.movies = pd.read_csv(
            os.path.join(data_dir, "movies.dat"),
            sep="::",
            engine="python",
            names=["movie_id", "title", "genres"],
            encoding="latin-1"
        )

        self.movies["genres"] = self.movies["genres"].fillna("")

        # Collaborative filtering
        reader = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(
            self.ratings[["user_id", "movie_id", "rating"]],
            reader
        )

        trainset = dataset.build_full_trainset()

        self.cf_model = SVD(
            n_factors=50,
            n_epochs=20,
            random_state=42
        )

        self.cf_model.fit(trainset)

        # Content-based model
        self.vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b[\w-]+\b"
        )

        self.content_matrix = self.vectorizer.fit_transform(
            self.movies["genres"].str.replace("|", " ", regex=False)
        )

        self.movie_to_index = {
            movie_id: i
            for i, movie_id in enumerate(self.movies["movie_id"])
        }

        # User interaction counts
        self.user_counts = (
            self.ratings.groupby("user_id")
            .size()
            .to_dict()
        )

        # Popularity
        self.popularity = (
            self.ratings.groupby("movie_id")
            .agg(
                rating_count=("rating", "count"),
                avg_rating=("rating", "mean")
            )
            .reset_index()
        )

        self.popularity["pop_score"] = (
            self.popularity["avg_rating"]
            * np.log1p(self.popularity["rating_count"])
        )

        self.popularity_scores = dict(
            zip(
                self.popularity["movie_id"],
                self.popularity["pop_score"]
            )
        )

    def get_weight(self, user_id):
        """
        Dynamic blending:
        0 interactions -> 0% CF
        1-4 interactions -> mostly content
        More interactions -> increasingly trust CF
        """
        count = self.user_counts.get(user_id, 0)

        cf_weight = min(count / 20.0, 0.9)
        content_weight = 1.0 - cf_weight

        return cf_weight, content_weight

    def content_score(self, user_id, movie_id):
        if movie_id not in self.movie_to_index:
            return 0.0

        user_history = self.ratings[
            self.ratings["user_id"] == user_id
        ]

        if user_history.empty:
            return 0.0

        target_index = self.movie_to_index[movie_id]

        scores = []

        for _, row in user_history.iterrows():
            liked_movie = row["movie_id"]

            if liked_movie not in self.movie_to_index:
                continue

            liked_index = self.movie_to_index[liked_movie]

            similarity = cosine_similarity(
                self.content_matrix[target_index],
                self.content_matrix[liked_index]
            )[0][0]

            # Give higher-rated movies more influence
            scores.append(similarity * (row["rating"] / 5.0))

        if not scores:
            return 0.0

        return float(np.mean(scores))

    def collaborative_score(self, user_id, movie_id):
        try:
            prediction = self.cf_model.predict(
                user_id,
                movie_id
            )
            return float(prediction.est)
        except Exception:
            return 0.0

    def recommend(self, user_id, k=10):
        watched = set(
            self.ratings[
                self.ratings["user_id"] == user_id
            ]["movie_id"]
        )

        cf_weight, content_weight = self.get_weight(user_id)

        candidates = []

        for movie_id in self.movies["movie_id"]:
            if movie_id in watched:
                continue

            cf = self.collaborative_score(
                user_id,
                movie_id
            )

            content = self.content_score(
                user_id,
                movie_id
            )

            # Normalize rating prediction to 0-1
            cf_norm = (cf - 1.0) / 4.0

            hybrid = (
                cf_weight * cf_norm
                + content_weight * content
            )

            candidates.append(
                (movie_id, hybrid, cf_norm, content)
            )

        candidates.sort(
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for movie_id, hybrid, cf, content in candidates[:k]:
            movie = self.movies[
                self.movies["movie_id"] == movie_id
            ].iloc[0]

            results.append({
                "movie_id": int(movie_id),
                "title": movie["title"],
                "genres": movie["genres"],
                "hybrid_score": round(float(hybrid), 4),
                "cf_score": round(float(cf), 4),
                "content_score": round(float(content), 4),
                "cf_weight": round(float(cf_weight), 2),
                "content_weight": round(float(content_weight), 2)
            })

        return results
