# Movie Recommender Research App

A research-driven movie recommender that tests whether SVD matrix factorization and KNN collaborative filtering improve rating prediction and top-k recommendation quality under a time-based validation split.

## Why This Project Matters

This project presents a complete recommender-systems study rather than a model-only experiment. Starting from MovieLens 100k ratings, it defines a research question, tests three hypotheses, compares SVD and KNNWithMeans against a popularity baseline, and evaluates the models with a time-based split that trains on past ratings and tests on future ratings. The results show a realistic tradeoff: SVD reduced RMSE by 7.4% and MAE by 12.5% compared with the baseline, but popularity still won the offline top-k ranking metrics. This makes the project academically stronger because it does not claim that one model wins everything; it explains why rating prediction and recommendation ranking must be evaluated separately.

## Research Question

Can matrix factorization and neighborhood collaborative filtering improve both rating prediction and top-k recommendation quality under realistic temporal validation?

## Hypotheses And Findings

| Hypothesis | Claim Tested | Result | Interpretation |
| --- | --- | --- | --- |
| H1 | SVD and KNNWithMeans reduce RMSE and MAE compared with a global-mean baseline. | Supported | SVD was the strongest rating predictor, reducing RMSE by 7.4% and MAE by 12.5%. |
| H2 | SVD and KNNWithMeans improve precision@10, recall@10, and nDCG@10 compared with popularity ranking. | Not supported | Popularity won the top-k metrics, showing that better rating prediction does not automatically produce better top-10 lists. |
| H3 | Time-based validation gives a more realistic estimate than a random split. | Supported as the main evaluation design | The pipeline trains on earlier ratings and tests on later ratings, avoiding look-ahead bias. |

## Methodology

Dataset: MovieLens 100k, using the original user IDs, movie IDs, ratings, movie titles, and timestamps.

Validation: five-fold expanding-window temporal validation. Each fold trains on earlier ratings and tests on the next chronological block.

Positive relevance definition: a held-out movie is relevant if the user rated it `>= 4`.

Models compared:

- Popularity/global-mean baseline: reference model for rating prediction, top-k ranking, and cold-start fallback.
- KNNWithMeans: neighborhood collaborative filtering from Surprise.
- SVD: matrix factorization from Surprise and the final served personalized model.

Metrics:

- Rating prediction: RMSE and MAE.
- Top-k ranking: precision@10, recall@10, nDCG@10.
- Catalog behavior: coverage@10.

## Project Architecture

```text
MovieLens 100k
    |
    v
Data download + ETL
    |
    v
Clean ratings with timestamps
    |
    +--> Research pipeline
    |       |-- Popularity baseline
    |       |-- KNNWithMeans
    |       |-- SVD
    |       |-- Time-based expanding-window validation
    |       |-- RMSE, MAE, precision@10, recall@10, nDCG@10, coverage@10
    |       +--> reports/model_comparison*.csv + MLflow logs
    |
    +--> Production training
            |-- Final SVD trained on all ratings
            +--> models/svd_model.pkl + metadata
                    |
                    v
              FastAPI recommendation API
                    |
                    v
              Streamlit research interface
```

## Exact Results

Five-fold expanding-window temporal validation on MovieLens 100k:

| Model | RMSE | MAE | Precision@10 | Recall@10 | nDCG@10 | Coverage@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularity baseline | 1.1317 | 0.9474 | 0.1978 | 0.0581 | 0.1931 | 0.0268 |
| KNNWithMeans | 1.1146 | 0.9251 | 0.0845 | 0.0178 | 0.0972 | 0.0574 |
| SVD | 1.0485 | 0.8288 | 0.1707 | 0.0456 | 0.1614 | 0.0822 |

Key insight: SVD was the best rating predictor and recommended a broader part of the catalog, with about 3.1x the coverage of popularity. Popularity remained strongest for offline top-k ranking. The system therefore serves SVD for known users, while keeping popularity as an explicit cold-start fallback.

## Reproducible Research Output

Run the full model comparison:

```bash
make evaluate
```

Canonical outputs:

- `reports/model_comparison.csv`: fold-level metrics.
- `reports/model_comparison_summary.csv`: mean and standard deviation by model.
- MLflow experiment: `MovieRecommender_ResearchComparison`.

## Reproduction And Serving

Local Python workflow:

```bash
make install
make data
make evaluate
make train
make api
```

In a second terminal, start the Streamlit interface:

```bash
make ui
```

Docker workflow:

```bash
make docker-build
make docker-setup
make docker-up
```

Then open:

- API docs: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501

Fast verification path:

```bash
make check
make evaluate-smoke
```

Example endpoints:

```bash
curl "http://localhost:8000/health"
curl "http://localhost:8000/recommendations?user_id=196&k=10&exclude_seen=true"
curl "http://localhost:8000/recommendations?user_id=9999&k=10&exclude_seen=true"
```

User `196` demonstrates personalized SVD recommendations. User `9999` demonstrates the cold-start popularity fallback.

## Streamlit Interface

- Personalized SVD recommendations for known MovieLens users.
- Cold-start popularity fallback for unknown users.
- User profile and rating history for the selected user.
- Research results panel showing the tradeoff between rating prediction and top-k ranking.
- API latency and whether already-rated movies were excluded.

## Documentation

Recommended order for reviewing the research and implementation:

- `docs/research_report.md`: main research report, including motivation, hypotheses, methodology, results, interpretation, implementation, and limitations.
- `docs/model_card.md`: formal summary of dataset, model behavior, evaluation, cold start, limitations, and ethical notes.
- `notebooks/01_temporal_evaluation_analysis.ipynb`: dataset and temporal-evaluation analysis notebook.
- `notebooks/02_serving_and_fallback_analysis.ipynb`: serving behavior, user-history, and fallback analysis notebook.

## Repository Guide

- `src/compare_models.py`: full research comparison pipeline.
- `src/evaluation.py`: temporal splits and ranking metrics.
- `src/train_svd.py`: production SVD artifact training.
- `src/recommender_service.py`: recommendation and fallback logic.
- `src/api.py`: FastAPI serving layer.
- `src/streamlit_app.py`: interactive Streamlit interface.
- `reports/model_comparison.csv`: fold-level research results.
- `reports/model_comparison_summary.csv`: summary metrics.
- `docs/research_report.md`: research report.
- `docs/model_card.md`: model card.
- `Dockerfile` and `docker-compose.yml`: containerized API and Streamlit interface.

## Limitations

- MovieLens 100k is small and old, so the results are best understood as an applied ML study rather than a production benchmark.
- Offline top-k metrics approximate recommendation quality but do not directly measure user satisfaction.
- The deployed model learns from historical explicit ratings, not live feedback.
- No content-based features are used, so new movies cannot be personalized until they receive ratings.
- The cold-start user path falls back to popularity rather than asking for onboarding preferences.

## Next Steps

- Add a hybrid content-based fallback using genre metadata.
- Add diversity and novelty metrics alongside accuracy metrics.
- Add a simple onboarding flow for new users.
- Add screenshots of the final Streamlit interface.
- Add CI so tests and smoke evaluation run automatically on pull requests.

## Testing

```bash
make check
```

The test suite covers temporal split ordering, rating metrics, top-k metrics, seen-item exclusion, cold-start fallback, and API response behavior.
