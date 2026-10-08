# Hybrid Recommendation Engine with Cold-Start Handling

An AI/ML movie recommendation system combining collaborative filtering and content-based recommendation with dynamic weighting for sparse-user scenarios.

## Features
- Collaborative filtering using SVD
- Content-based filtering using movie genres
- Dynamic cold-start-aware weighting
- Precision@10, Recall@10 and NDCG@10 evaluation
- FastAPI backend
- React frontend

## Results

| Segment | Precision@10 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| Simulated Cold Start (4 interactions) | 0.00325 | 0.0325 | 0.01403 |
| Warm Users | 0.00125 | 0.0125 | 0.00478 |

## Cold-Start Evaluation

MovieLens 1M does not contain users with fewer than five total interactions. Therefore, cold-start behavior was simulated by exposing only four historical interactions and evaluating against a held-out interaction.

## Architecture

MovieLens 1M -> SVD + Genre Similarity -> Dynamic Weighting -> Hybrid Ranking -> FastAPI -> React

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Scikit-Surprise, FastAPI, React, Vite.

## Run Backend

cd backend
source venv/bin/activate
uvicorn app:app --reload

## Run Frontend

cd frontend
npm install
npm run dev

## Future Improvements

- Richer metadata and embeddings
- Diversity-aware re-ranking
- True item cold-start experiments
- AWS deployment
- Popularity and collaborative-filtering baselines
