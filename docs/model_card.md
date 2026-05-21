# Model Card: Movie Recommender SVD Service

## Model Overview

This model card describes the SVD recommender served by the project API. The service returns personalized movie recommendations for known MovieLens users and a popularity-based fallback for unknown users.

- Model type: Matrix factorization with Surprise SVD.
- Production artifact: `models/svd_model.pkl`.
- Metadata artifact: `models/svd_model_metadata.json`.
- Model version: `svd-n100-e20`.
- Fallback model: Smoothed popularity ranking.

## Dataset

The project uses MovieLens 100k.

- Ratings: 100,000 explicit ratings.
- Users: 943.
- Movies: 1,682.
- Rating scale: 1 to 5 stars.
- Time information: Unix timestamps converted into datetime features.

The model is trained on historical explicit ratings. Movie titles are used for display only; the SVD model itself uses user IDs, item IDs, and ratings.

## Intended Use

This recommender is intended for:

- Demonstrating a complete recommender-systems research workflow.
- Comparing collaborative filtering methods under time-based validation.
- Serving movie recommendations in a local research application.
- Showing how evaluation findings affect API and fallback design.

This model is not intended for:

- Production entertainment platforms.
- High-stakes decision-making.
- Personalized recommendations for real users outside MovieLens-style rating data.
- Recommending newly released movies with no ratings.

## Evaluation

The project evaluates models with five-fold expanding-window temporal validation. Each fold trains on earlier ratings and tests on later ratings to avoid look-ahead bias.

Metrics:

- Rating prediction: RMSE, MAE.
- Top-k recommendation: precision@10, recall@10, nDCG@10.
- Catalog behavior: coverage@10.

Summary results:

| Model | RMSE | MAE | Precision@10 | Recall@10 | nDCG@10 | Coverage@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularity baseline | 1.1317 | 0.9474 | 0.1978 | 0.0581 | 0.1931 | 0.0268 |
| KNNWithMeans | 1.1146 | 0.9251 | 0.0845 | 0.0178 | 0.0972 | 0.0574 |
| SVD | 1.0485 | 0.8288 | 0.1707 | 0.0456 | 0.1614 | 0.0822 |

Interpretation:

- SVD reduced RMSE by 7.4% and MAE by 12.5% compared with the popularity/global-mean baseline.
- Popularity achieved the best offline precision@10, recall@10, and nDCG@10.
- SVD achieved about 3.1x the catalog coverage of popularity.
- The result shows that better rating prediction does not automatically imply better top-k recommendation quality.

## Cold-Start Behavior

For known users, the API scores candidate movies with the trained SVD model and excludes already-rated movies by default.

For unknown users, the API sets `fallback_used=true` and returns smoothed popularity recommendations. This avoids pretending that personalized recommendations exist when the model has no user history.

The API response includes:

- `model_name`
- `model_version`
- `fallback_used`
- `latency_ms`
- ranked recommendations with `movie_id`, `title`, `score`, and `reason`

## Limitations

- MovieLens 100k is small and from an older movie-rating context.
- Offline metrics do not directly measure real user satisfaction.
- The model only learns from explicit ratings, not watch behavior, search behavior, or feedback loops.
- New users receive popularity recommendations until they have rating history.
- New movies cannot be personalized until rating data exists.
- Popularity-based fallback can over-recommend already famous or mainstream movies.
- SVD latent factors are not directly interpretable without additional analysis.

## Ethical Notes

- Time-based validation avoids look-ahead bias and inflated performance claims.
- The service explicitly marks cold-start fallback behavior instead of presenting it as personalization.
- The project reports the unsupported H2 result rather than claiming all metrics improved.
- Recommendation systems can amplify popularity bias; this is visible in the strong top-k performance of the popularity baseline.
- Because the dataset is historical and anonymized, the interface should be treated as a research artifact rather than a representation of real current user preferences.

## Monitoring And Future Improvements

For a real deployment, the next monitoring and improvement steps would be:

- Track latency and fallback rate.
- Monitor recommendation diversity and novelty.
- Add content-based features for new movies.
- Add onboarding preferences for new users.
- Evaluate with online feedback or user studies instead of offline metrics only.
