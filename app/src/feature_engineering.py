# feature_engineering.py
import os
import pickle
import numpy as np
import pandas as pd
import logging
from .path_config import TFIDF_VECTORIZER_PATH

logger = logging.getLogger(__name__)

def extract_candidate_features_improved(candidate, model_data, debug=False):
    """
    Enhanced feature extraction with minimal debug output.
    Maintains original feature engineering logic while handling new location context.
    """
    try:
        if debug:
            logger.info(f"Loading TF-IDF vectorizer from: {TFIDF_VECTORIZER_PATH}")

        with open(TFIDF_VECTORIZER_PATH, 'rb') as f:
            tfidf = pickle.load(f)

        # Extract basic features (maintaining original logic)
        features = {
            "review_count": candidate.get("user_ratings_total", 0),
            "price_range": candidate.get("price_level", 2),  # Default to 2 if missing
            "is_open": 1,  # Google Places usually returns open places
            "review_useful": 0,  # Placeholder for Yelp-specific features
            "review_funny": 0,
            "review_cool": 0,
            "review_engagement": candidate.get("user_ratings_total", 0) * 0.1,
            "popularity_score": np.log1p(candidate.get("user_ratings_total", 0)),
            "latitude": candidate.get("geometry", {}).get("location", {}).get("lat", 0),
            "longitude": candidate.get("geometry", {}).get("location", {}).get("lng", 0),
            "location_cluster": 0
        }

        # Process reviews and business types
        review_text = ""
        if "processed_reviews" in candidate:
            reviews = candidate["processed_reviews"]
            review_text = " ".join([review.get("text", "") for review in reviews])

        types_text = " ".join(candidate.get("types", []))
        combined_text = f"{review_text} {types_text}"

        # Transform text using TF-IDF (maintaining original logic)
        if combined_text.strip():
            if debug:
                logger.info("Processing text features")
            text_features = tfidf.transform([combined_text])
            text_feature_names = [f'text_feature_{i}' for i in range(text_features.shape[1])]
            for i, value in enumerate(text_features.toarray()[0]):
                features[text_feature_names[i]] = value
        else:
            if debug:
                logger.info("No text content available, using zero features")
            for i in range(100):  # Assuming 100 text features from training
                features[f'text_feature_{i}'] = 0.0

        # Create DataFrame from features
        candidate_features = pd.DataFrame([features])

        # Ensure all model features are present
        missing_cols = set(model_data['feature_names']) - set(candidate_features.columns)
        for col in missing_cols:
            candidate_features[col] = 0

        # Reorder columns to match training data
        candidate_features = candidate_features[model_data['feature_names']]

        if debug:
            logger.info(f"Generated {len(candidate_features.columns)} features")
            non_zero = candidate_features.iloc[0][candidate_features.iloc[0] != 0]
            logger.info(f"Non-zero features: {len(non_zero)}")

        return candidate_features

    except Exception as e:
        logger.error(f"Error in feature extraction: {e}")
        # Return DataFrame with zeros as fallback
        return pd.DataFrame([[0] * len(model_data['feature_names'])],
                          columns=model_data['feature_names'])

def predict_scores(candidates_features, model_data, debug=False):
    """
    Use the model to predict scores with minimal debugging output.
    Maintains original scoring logic.
    """
    if debug:
        logger.info(f"Predicting scores for {len(candidates_features)} candidates")

    # Get raw predictions
    raw_scores = model_data['model'].predict(candidates_features)

    # Scale scores (only if there is variation)
    if raw_scores.max() > raw_scores.min():
        scaled_scores = 20 + (raw_scores - raw_scores.min()) * 60 / (raw_scores.max() - raw_scores.min())
    else:
        scaled_scores = np.full_like(raw_scores, 50.0)

    if debug:
        logger.info(f"Score range: {scaled_scores.min():.1f} - {scaled_scores.max():.1f}")

    return scaled_scores
