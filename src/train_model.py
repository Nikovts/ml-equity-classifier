import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report

def characterize_stock_regime(train_df):
    """
    META-CLASSIFIER LAYER v5.5:
    Evaluates raw historical P/E multiples to isolate true secular growth assets.
    """
    print("\n🔍 RUNNING ALGORITHMIC REGIME CHARACTERIZATION...")
    
    velocity_autocorr = train_df['feature_trend_20d_velocity'].autocorr(lag=5)
    avg_volatility = train_df['volatility_20d'].mean()
    
    if 'raw_ratio_pe' in train_df.columns:
        historical_median_pe = train_df['raw_ratio_pe'].median()
    else:
        historical_median_pe = train_df['ratio_pe'].median()
        print("⚠️ Warning: raw_ratio_pe not found. Operating on transformed array.")
    
    print(f"-> 5-Day Velocity Autocorrelation: {velocity_autocorr:.4f}")
    print(f"-> Average Historical Baseline Volatility: {avg_volatility:.4f}")
    print(f"-> True Historical Valuation Baseline (Median P/E): {historical_median_pe:.2f}")
    
    has_growth_premium = historical_median_pe >= 22.0
    is_high_vol = avg_volatility >= 0.022
    is_growth_vol = has_growth_premium and (avg_volatility >= 0.017)
    
    if (is_high_vol or is_growth_vol) and velocity_autocorr >= -0.02:
        print("🎯 REGIME DIAGNOSIS: TYPE-A [High-Reflexivity Breakout Engine]")
        return "TYPE_A_MOMENTUM"
    elif velocity_autocorr < -0.02 and (is_high_vol or is_growth_vol):
        print("⚡ REGIME DIAGNOSIS: TYPE-C [High-Volatility / Secular Growth Reflexive Value Engine]")
        return "TYPE_C_REFLEXIVE_VALUE"
    else:
        print("⚖️ REGIME DIAGNOSIS: TYPE-B [Mean-Reverting Stable Value Engine]")
        return "TYPE_B_MEAN_REVERSION"


def train_production_engine(identifier="Asset_X", train_path="data/train_matrix.csv", test_path="data/test_matrix.csv"):
    print(f"\n=============================================================")
    print(f"🔒 EXECUTING TICKER-AGNOSTIC PRODUCTION HARNESS v5.5: [{identifier}]")
    print(f"=============================================================")
    
    # Path safety check
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing data matrices. Please run the data pipeline engine first.")

    train_df = pd.read_csv(train_path, index_col="Date", parse_dates=True)
    test_df = pd.read_csv(test_path, index_col="Date", parse_dates=True)
    
    stock_character = characterize_stock_regime(train_df)
    
    # Map target: compresses Short (-1) and Neutral (0) into Skip (0) for Long-Only Alpha Strategy
    y_train = train_df['target'].apply(lambda x: 1 if x == 1 else 0)
    y_test = test_df['target'].apply(lambda x: 1 if x == 1 else 0)

    if stock_character == "TYPE_A_MOMENTUM":
        selected_features = ['ratio_pe', 'feature_trend_20d_velocity', 'feature_ratio_pe_velocity_5d', 'trend_60d_sector_spread', 'volatility_20d']
        X_train = train_df[selected_features].copy()
        X_test = test_df[selected_features].copy()
        param_grid = {
            'classifier__n_estimators': [150, 200],
            'classifier__max_depth': [6, 8],
            'classifier__max_features': [0.5],
            'classifier__min_samples_leaf': [1]
        }
        TARGET_BUY_PERCENTILE = 85.0

    elif stock_character == "TYPE_B_MEAN_REVERSION":
        selected_features = ['ratio_pe', 'feature_trend_20d_velocity', 'feature_ratio_pe_velocity_5d']
        X_train = train_df[selected_features].copy()
        X_test = test_df[selected_features].copy()
        
        pe_train_median = train_df['ratio_pe'].median()
        X_train['feature_pe_distance_from_median'] = train_df['ratio_pe'] - pe_train_median
        X_test['feature_pe_distance_from_median'] = test_df['ratio_pe'] - pe_train_median
        
        param_grid = {
            'classifier__n_estimators': [150, 200],
            'classifier__max_depth': [4, 5],
            'classifier__max_features': [0.5, 0.6],
            'classifier__min_samples_leaf': [12, 18]
        }
        TARGET_BUY_PERCENTILE = 80.0

    elif stock_character == "TYPE_C_REFLEXIVE_VALUE":
        selected_features = ['ratio_pe', 'feature_trend_20d_velocity', 'feature_ratio_pe_velocity_5d', 'trend_60d_sector_spread', 'volatility_20d']
        X_train = train_df[selected_features].copy()
        X_test = test_df[selected_features].copy()
        
        pe_train_median = train_df['ratio_pe'].median()
        X_train['feature_pe_distance_from_median'] = train_df['ratio_pe'] - pe_train_median
        X_test['feature_pe_distance_from_median'] = test_df['ratio_pe'] - pe_train_median
        
        param_grid = {
            'classifier__n_estimators': [150, 200],
            'classifier__max_depth': [5, 6, 7],
            'classifier__max_features': [0.5],
            'classifier__min_samples_leaf': [2, 4, 6]
        }
        TARGET_BUY_PERCENTILE = 83.0

    # Interaction feature engine
    X_train['feature_momentum_acceleration'] = np.where(
        (X_train['feature_trend_20d_velocity'] > 0) & (X_train['feature_ratio_pe_velocity_5d'] > 0),
        X_train['feature_trend_20d_velocity'] * X_train['feature_ratio_pe_velocity_5d'], 0.0
    )
    X_test['feature_momentum_acceleration'] = np.where(
        (X_test['feature_trend_20d_velocity'] > 0) & (X_test['feature_ratio_pe_velocity_5d'] > 0),
        X_test['feature_trend_20d_velocity'] * X_test['feature_ratio_pe_velocity_5d'], 0.0
    )
    
    for df_slice in [X_train, X_test]:
        if 'raw_ratio_pe' in df_slice.columns:
            df_slice.drop(columns=['raw_ratio_pe'], inplace=True)

    stock_pipeline = Pipeline(steps=[
        ('standard_scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced_subsample'))
    ])

    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=stock_pipeline, param_grid=param_grid, cv=cv_strategy, scoring='f1_macro', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_
    
    # Score Output Generation
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]
    dynamic_threshold = np.percentile(y_prob, TARGET_BUY_PERCENTILE)
    y_pred = (y_prob >= dynamic_threshold).astype(int)

    print(f"\n=== PERFORMANCE SUMMARY ({identifier} | {stock_character}) ===")
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['Skip (0)', 'Buy (1)'], zero_division=0))
    
    # Crucial architectural update: Return both pipeline and the operational threshold
    return best_pipeline, dynamic_threshold

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    train_production_engine('NVDA')