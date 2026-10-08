import { useState } from "react";
import "./index.css";

const API = "http://127.0.0.1:8000";

function App() {
  const [userId, setUserId] = useState("1");
  const [data, setData] = useState(null);
  const [userInfo, setUserInfo] = useState(null);
  const [loading, setLoading] = useState(false);

  const getRecommendations = async () => {
    setLoading(true);

    try {
      const [recResponse, userResponse] = await Promise.all([
        fetch(`${API}/recommend/${userId}?k=10`),
        fetch(`${API}/user/${userId}`)
      ]);

      setData(await recResponse.json());
      setUserInfo(await userResponse.json());
    } catch (error) {
      console.error(error);
      alert("Backend is not reachable. Make sure Terminal 1 is running.");
    }

    setLoading(false);
  };

  return (
    <div className="app">
      <header>
        <div className="badge">AI / ML • RECOMMENDATION ENGINE</div>

        <h1>
          Hybrid Movie
          <br />
          Recommendation Engine
        </h1>

        <p>
          Cold-start aware recommendations using collaborative filtering,
          content similarity, and dynamic model blending.
        </p>
      </header>

      <section className="control">
        <label>User ID</label>

        <div className="search">
          <input
            type="number"
            min="1"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
          />

          <button onClick={getRecommendations}>
            {loading ? "Loading..." : "Recommend"}
          </button>
        </div>
      </section>

      {userInfo && (
        <section className="stats">
          <div>
            <span>Interactions</span>
            <strong>{userInfo.interaction_count}</strong>
          </div>

          <div>
            <span>User Segment</span>
            <strong>
              {userInfo.cold_start ? "Cold Start" : "Warm"}
            </strong>
          </div>

          <div>
            <span>CF Weight</span>
            <strong>{(userInfo.cf_weight * 100).toFixed(0)}%</strong>
          </div>

          <div>
            <span>Content Weight</span>
            <strong>{(userInfo.content_weight * 100).toFixed(0)}%</strong>
          </div>
        </section>
      )}

      {data && (
        <section className="results">
          <div className="section-title">
            <h2>Recommended for User {data.user_id}</h2>
            <span>Top 10</span>
          </div>

          <div className="movies">
            {data.recommendations.map((movie, index) => (
              <article className="movie" key={movie.movie_id}>
                <div className="rank">
                  #{index + 1}
                </div>

                <div className="movie-info">
                  <h3>{movie.title}</h3>

                  <p>
                    {movie.genres.replaceAll("|", " • ")}
                  </p>

                  <div className="scores">
                    <span>
                      Hybrid <b>{movie.hybrid_score}</b>
                    </span>

                    <span>
                      CF <b>{movie.cf_score}</b>
                    </span>

                    <span>
                      Content <b>{movie.content_score}</b>
                    </span>
                  </div>
                </div>
              </article>
            ))}
          </div>
        </section>
      )}

      <footer>
        Hybrid Recommendation Engine • MovieLens 1M • Dynamic Cold-Start Blending
      </footer>
    </div>
  );
}

export default App;
