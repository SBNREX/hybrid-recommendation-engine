import os
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(BASE, "ml-1m")

ratings = pd.read_csv(
    os.path.join(DATA, "ratings.dat"),
    sep="::",
    engine="python",
    names=["user_id", "movie_id", "rating", "timestamp"],
    encoding="latin-1"
)

movies = pd.read_csv(
    os.path.join(DATA, "movies.dat"),
    sep="::",
    engine="python",
    names=["movie_id", "title", "genres"],
    encoding="latin-1"
)

np.random.seed(42)

user_counts = ratings.groupby("user_id").size()

eligible_users = user_counts[user_counts >= 25].index

sample_users = np.random.choice(
    eligible_users,
    min(400, len(eligible_users)),
    replace=False
)

def evaluate(mode, k=10):

    train_parts = []
    test_rows = []

    for user_id in sample_users:

        user_data = ratings[
            ratings.user_id == user_id
        ].sort_values("timestamp")

        if mode == "cold":
            # Simulate a new/sparse user:
            # expose only 4 historical interactions
            history = user_data.iloc[:4]
            test_row = user_data.iloc[4]

        else:
            # Warm user:
            # expose most history and hold out final interaction
            history = user_data.iloc[:-1]
            test_row = user_data.iloc[-1]

        train_parts.append(history)
        test_rows.append(test_row)

    other_users = ratings[
        ~ratings.user_id.isin(sample_users)
    ]

    train_df = pd.concat(
        [other_users] + train_parts,
        ignore_index=True
    )

    train_df = train_df[
        ["user_id", "movie_id", "rating"]
    ]

    reader = Reader(rating_scale=(1, 5))

    dataset = Dataset.load_from_df(
        train_df,
        reader
    )

    trainset = dataset.build_full_trainset()

    model = SVD(
        n_factors=50,
        n_epochs=20,
        random_state=42
    )

    model.fit(trainset)

    all_movies = set(movies.movie_id)

    precisions = []
    recalls = []
    ndcgs = []

    for row in test_rows:

        user_id = int(row.user_id)
        test_movie = int(row.movie_id)

        watched = set(
            train_df[
                train_df.user_id == user_id
            ].movie_id
        )

        candidates = list(
            all_movies - watched
        )

        predictions = []

        for movie_id in candidates:
            try:
                score = model.predict(
                    user_id,
                    int(movie_id)
                ).est

                predictions.append(
                    (movie_id, score)
                )
            except Exception:
                pass

        predictions.sort(
            key=lambda x: x[1],
            reverse=True
        )

        top_k = [
            movie_id
            for movie_id, _ in predictions[:k]
        ]

        hit = int(test_movie in top_k)

        precisions.append(hit / k)
        recalls.append(hit)

        if hit:
            rank = top_k.index(test_movie) + 1
            ndcgs.append(1 / np.log2(rank + 1))
        else:
            ndcgs.append(0.0)

    return (
        float(np.mean(precisions)),
        float(np.mean(recalls)),
        float(np.mean(ndcgs))
    )

print("Evaluating simulated cold-start users...")
cold_p, cold_r, cold_n = evaluate("cold")

print("Evaluating warm users...")
warm_p, warm_r, warm_n = evaluate("warm")

results = pd.DataFrame([
    {
        "segment": "Simulated Cold Start (4 interactions)",
        "Precision@10": cold_p,
        "Recall@10": cold_r,
        "NDCG@10": cold_n
    },
    {
        "segment": "Warm Users",
        "Precision@10": warm_p,
        "Recall@10": warm_r,
        "NDCG@10": warm_n
    }
])

results.to_csv(
    os.path.join(BASE, "evaluation_results.csv"),
    index=False
)

print("\n===== FINAL EVALUATION RESULTS =====")
print(results.to_string(index=False))
print("\nSaved: evaluation_results.csv")
