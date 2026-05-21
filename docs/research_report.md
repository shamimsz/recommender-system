# Research Narrative: Temporal Evaluation Of Collaborative Filtering For Movie Recommendation

This project designs, implements, and evaluates a MovieLens recommender system around a focused research question: do collaborative filtering models improve both rating prediction and top-k recommendation quality when the evaluation respects time? The experimental pipeline includes SVD matrix factorization, KNNWithMeans neighborhood collaborative filtering, a popularity/global-mean baseline, expanding-window temporal validation, rating metrics, ranking metrics, MLflow logging, and a served API and Streamlit interface. The results show the central finding of the project: SVD reduced RMSE by 7.4% and MAE by 12.5% compared with the baseline, but popularity still achieved the best precision@10, recall@10, and nDCG@10. That tradeoff is the research contribution of this project.

## 1. Research Motivation

Recommendation systems are often presented as a simple prediction problem: estimate how much a user will like an item. In practice, a recommender also has to rank a small list of items for the user to see first. These two goals can disagree. A model can predict ratings accurately while producing a top-10 list that performs worse than a simple popularity ranking.

The project studies that gap directly. Instead of asking only whether SVD can predict ratings, the evaluation tests whether collaborative filtering improves two different outcomes:

- Rating prediction: how close are predicted ratings to held-out ratings?
- Top-k recommendation: how good is the ranked list of recommended movies?

The evaluation uses a time-based split because recommendation is naturally temporal. A real system trains on past behavior and recommends future items. A random split can leak future preferences into training and make the evaluation too optimistic.

## 2. Dataset And Scope

MovieLens 100k is the only dataset used in this project. It contains:

- 100,000 explicit ratings.
- 943 users.
- 1,682 movies.
- Ratings from 1 to 5.
- Unix timestamps for every rating.

The scope is intentionally focused: no external APIs, poster services, deep learning models, or additional datasets are added. This keeps the research question clean: how do standard collaborative filtering methods behave on explicit rating data when evaluated with time-aware validation and both prediction and ranking metrics?

The ETL step in `src/etl.py` converts the raw MovieLens files into one shared cleaned dataset:

```text
data/cleaned/cleaned_ratings_with_time.csv
```

That file feeds the research script, notebooks, final model training, FastAPI service, and Streamlit interface.

## 3. Research Question

Can matrix factorization and neighborhood collaborative filtering improve both rating prediction and top-k recommendation quality under realistic temporal validation?

This question is specific on purpose:

- Matrix factorization means SVD.
- Neighborhood collaborative filtering means KNNWithMeans.
- Rating prediction means RMSE and MAE.
- Top-k quality means precision@10, recall@10, nDCG@10, and coverage@10.
- Realistic validation means training on earlier ratings and testing on later ratings.

## 4. Hypotheses

The project tests three hypotheses because it makes three separate claims.

### H1: Rating Prediction

SVD and KNNWithMeans will reduce RMSE and MAE compared with a global-mean popularity baseline.

Rationale: collaborative filtering can use user-item interaction patterns that a global average ignores. SVD should perform especially well because latent factors can represent hidden preference structure in the sparse rating matrix.

### H2: Top-k Recommendation Quality

SVD and KNNWithMeans will improve precision@10, recall@10, and nDCG@10 compared with popularity-based recommendations.

Rationale: a personalized model should, in theory, rank user-relevant movies above generally popular movies.

### H3: Evaluation Method

Time-based validation gives a more realistic estimate than a random split because it avoids look-ahead bias.

Rationale: future ratings should not be available when predicting future behavior. This is a methodological hypothesis: it defines the primary evaluation setting.

## 5. Baseline Design

The comparison starts with a popularity/global-mean baseline because a research comparison needs a serious reference point.

The baseline has two roles:

- For rating prediction, it predicts the global average rating.
- For top-k recommendation, it ranks movies by smoothed average rating.

The smoothing matters. A movie with one 5-star rating should not automatically outrank a movie with hundreds of strong ratings. The smoothed score makes popularity a stronger and fairer baseline.

This baseline also became useful in the application: for unknown users, the API falls back to popularity instead of pretending it can personalize without user history.

## 6. Models Compared

### Popularity Baseline

This model controls for the fact that famous or highly rated movies are often strong recommendations even without personalization.

### KNNWithMeans

KNNWithMeans is a neighborhood collaborative filtering model from Surprise. It is used instead of KNNBasic because it accounts for mean rating differences between users or items. That matters because some users rate strictly and others rate generously.

### SVD

SVD is the main proposed model. It factorizes the sparse user-item rating matrix into lower-dimensional user and movie representations. These latent factors are not directly labeled, but they can capture hidden preference patterns.

The final Streamlit interface serves SVD because it achieved the strongest rating prediction performance and the best catalog coverage.

## 7. Evaluation Design

The main research code is in:

```text
src/evaluation.py
src/compare_models.py
```

The full comparison runs with:

```bash
python -m src.compare_models
```

The pipeline follows these steps:

1. Load the cleaned MovieLens ratings.
2. Sort all ratings by timestamp.
3. Build five expanding-window temporal folds.
4. Train each model only on earlier ratings.
5. Test on the next chronological block.
6. Compute RMSE and MAE for held-out rating prediction.
7. Generate top-10 recommendations for users in the test fold.
8. Exclude movies the user already rated in training.
9. Treat held-out ratings `>= 4` as relevant.
10. Compute precision@10, recall@10, nDCG@10, and coverage@10.
11. Log fold-level and summary metrics to MLflow.
12. Save canonical results to `reports/`.

The main result files are:

```text
reports/model_comparison.csv
reports/model_comparison_summary.csv
```

## 8. Metrics

RMSE measures rating prediction error and penalizes large mistakes strongly.

MAE measures average absolute rating error in rating-point units.

Precision@10 measures how many of the 10 recommended movies were relevant.

Recall@10 measures how many relevant held-out movies appeared in the top 10.

nDCG@10 rewards relevant movies appearing near the top of the list.

Coverage@10 measures how much of the movie catalog appears across recommendation lists. This matters because a recommender that always suggests the same famous movies may be accurate but narrow.

## 9. Results

Five-fold expanding-window temporal validation produced these mean metrics:

| Model | RMSE | MAE | Precision@10 | Recall@10 | nDCG@10 | Coverage@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularity baseline | 1.1317 | 0.9474 | 0.1978 | 0.0581 | 0.1931 | 0.0268 |
| KNNWithMeans | 1.1146 | 0.9251 | 0.0845 | 0.0178 | 0.0972 | 0.0574 |
| SVD | 1.0485 | 0.8288 | 0.1707 | 0.0456 | 0.1614 | 0.0822 |

## 10. Hypothesis Evaluation

### H1: Supported

SVD achieved the best rating prediction performance:

- RMSE improved from 1.1317 to 1.0485, a 7.4% reduction.
- MAE improved from 0.9474 to 0.8288, a 12.5% reduction.

KNNWithMeans also improved over the baseline, but the gain was smaller. This supports the claim that collaborative filtering captures signal beyond the global mean, with SVD providing the strongest improvement.

### H2: Not Supported

Popularity achieved the best top-k ranking metrics:

- Precision@10: popularity 0.1978 vs SVD 0.1707.
- Recall@10: popularity 0.0581 vs SVD 0.0456.
- nDCG@10: popularity 0.1931 vs SVD 0.1614.

This result matters because it contradicts the simple expectation that better personalization automatically improves top-k recommendation. In this offline setting, popular movies are more likely to appear in users' held-out future ratings, so the popularity baseline remains difficult to beat.

### H3: Supported As The Primary Evaluation Design

The project uses time-based validation as the headline result. Each fold trains on earlier ratings and tests on later ratings, so the model never learns from the future. This makes the evaluation more realistic than a random split for a recommender-system setting.

## 11. Interpretation

The results show a clear tradeoff.

SVD is the best model for rating prediction. It also has the highest coverage@10: 0.0822 compared with 0.0268 for popularity, or about 3.1x higher catalog coverage. This means SVD recommends a broader set of movies instead of concentrating on the same small group of popular titles.

Popularity is the best model for the current offline top-k metrics. This does not mean popularity is always the best recommender; it means that in MovieLens 100k, under this held-out temporal evaluation, generally liked movies are very likely to match future rated items.

The final system design uses both findings:

- SVD serves known users because it is the strongest personalized rating model.
- Popularity serves unknown users because it is strong, simple, and honest for cold start.

## 12. From Research To Application

After the comparison, the final SVD model is trained on all MovieLens 100k ratings:

```bash
make train
```

This creates reproducible local artifacts:

```text
models/svd_model.pkl
models/svd_model_metadata.json
```

The API loads this SVD artifact and uses the popularity model as a fallback. The app therefore reflects the research results instead of treating the research and product sides as separate work.

## 13. API And Service Behavior

The recommendation logic is in `src/recommender_service.py`, and the FastAPI app is in `src/api.py`.

Main endpoint:

```text
GET /recommendations?user_id=196&k=10&exclude_seen=true
```

For known users, the service scores unseen movies with SVD and excludes already-rated movies by default.

For unknown users, the service returns popularity recommendations and sets `fallback_used=true`. This makes the cold-start behavior explicit.

## 14. Streamlit Interface

The Streamlit app makes the research result inspectable:

- It shows recommendations for a selected user.
- It displays whether the result came from SVD or popularity fallback.
- It shows user history so the recommendation context is concrete.
- It includes a research panel with the final metrics.
- It highlights the main conclusion: SVD wins rating prediction, while popularity wins top-k ranking.

## 15. Docker And Reproducibility

Docker makes the project easier to run on another machine:

```bash
make docker-build
make docker-setup
make docker-up
```

The setup service downloads data, runs ETL, and trains the model. The API and UI then use the generated data and model artifacts.

The Docker Compose configuration syntax was validated in this environment, but a full Docker image build has not yet been run here.

## 16. Skills Demonstrated

This project demonstrates the following capabilities:

- Turn an informal idea into a precise research question.
- State hypotheses and test them with measurable evidence.
- Implement time-based validation instead of relying on a random split.
- Compare advanced models against a meaningful baseline.
- Interpret a mixed result instead of forcing an overly simple conclusion.
- Connect research findings to API behavior, fallback design, and a usable UI.
- Document the methodology, limitations, and next research steps clearly.

## 17. Limitations

- MovieLens 100k is small and old.
- Offline metrics approximate user satisfaction but do not measure it directly.
- The model uses explicit ratings only, not implicit behavior such as watch time.
- New users need popularity fallback until they provide preferences.
- New movies cannot be personalized until they have ratings.
- The model does not use genres, plot text, or content embeddings.
- The app is a local research application, not a production deployment.

## 18. Next Research Steps

Planned research improvements include:

- Add diversity and novelty metrics to test whether SVD's higher coverage leads to better discovery.
- Add a genre-based hybrid model for cold-start movies.
- Compare temporal validation with a random split as a methodological appendix.
- Add an onboarding experiment where a new user rates a few movies before receiving recommendations.
- Add confidence intervals or paired statistical tests across folds.

## 19. Reproduction Commands

Prepare data:

```bash
make data
```

Run the full research comparison:

```bash
make evaluate
```

Train the served model:

```bash
make train
```

Run tests:

```bash
make check
```

Start the API and UI:

```bash
make api
make ui
```
