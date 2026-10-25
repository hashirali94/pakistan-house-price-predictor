# Pakistan House Price Predictor

Machine learning model that predicts house sale prices in 5 Pakistani cities: Karachi, Lahore, Islamabad, Rawalpindi and Faisalabad.

## Data
Zameen.com listings (Kaggle: "House Prices 2023 Dataset"), cleaned to 70,630 sale listings.

## Approach
- Removed rent listings and invalid rows (zero area, unrealistic prices)
- Features: city, location, property type, bedrooms, bathrooms, area (marla)
- Target: log of price
- Model: Histogram Gradient Boosting Regressor

## Results (20% hold-out test set)
- R2 score: 0.855
- Mean absolute error: about PKR 24.4 lakh

## Files
- notebookb8172fec43.ipynb: data cleaning, training and evaluation
- house_price_model.pkl: trained model
- Cleaned_data_for_model.csv: cleaned dataset

## Limitations
Based on listing (asking) prices, not final sale prices. Locations are encoded simply, and prices change over time.

## Demo app
Try it live: https://pakistan-house-price-predictor-otzdbywklfu5qcnpdq9eso.streamlit.app/

The app trains the model on the cleaned dataset when it first loads (about 30 seconds), then predicts prices for any city, area, property type, size, and number of rooms. It also shows how many similar listings the estimate is based on, and warns when data is thin.
