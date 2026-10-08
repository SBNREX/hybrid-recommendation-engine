from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from recommender import HybridRecommender

app = FastAPI(
    title="Hybrid Recommendation Engine",
    description="Cold-start aware movie recommendation system"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

print("Loading recommendation model...")
recommender = HybridRecommender()
print("Model loaded successfully.")


@app.get("/")
def root():
    return {
        "project": "Hybrid Recommendation Engine",
        "status": "running"
    }


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = 10):
    return {
        "user_id": user_id,
        "recommendations": recommender.recommend(user_id, k)
    }


@app.get("/user/{user_id}")
def user_info(user_id: int):
    count = recommender.user_counts.get(user_id, 0)
    cf_weight, content_weight = recommender.get_weight(user_id)

    return {
        "user_id": user_id,
        "interaction_count": count,
        "cold_start": count < 5,
        "cf_weight": cf_weight,
        "content_weight": content_weight
    }
