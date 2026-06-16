import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# =====================================================================
# CONFIGURATION PANEL: TOGGLE ENGINE & TARGET LAYERS HERE
# =====================================================================
MODEL_TYPE = 'random_forest'  # Options: 'logistic_regression' or 'random_forest'
COLLAPSE_TO_BINARY = True    # True: High-Conviction Binary, False: Baseline 3-Class

def inject_advanced_features(X):
    """
    Injects technical logic states and first derivatives (velocity).
    """
    X = X.copy()
    
    # 1. Technical logic states 
    if 'momentum_rsi' in X.columns:
        X['rsi_oversold'] = (X['momentum_rsi'] < 30).astype(int)
        X['rsi_overbought'] = (X['momentum_rsi'] > 70).astype(int)
        
    if 'trend_20d' in X.columns and 'trend_60d' in X.columns:
        X['trend_bullish_cross'] = (X['trend_20d'] > X['trend_60d']).astype(int)
        X['trend_bearish_cross'] = (X['trend_20d'] < X['trend_60d']).astype(int)
        
    # 2. First Derivatives / Velocity (5-day momentum shifts)
    features_to_differentiate = ['momentum_rsi', 'ratio_pe', 'ratio_pb', 'ratio_dcf_value']
    for col in features_to_differentiate:
        if col in X.columns:
            X[f'{col}_velocity_5d'] = X[col].diff(5).fillna(0)
            
    return X

def train_optimized_pipeline():
    print(f"=== INITIALIZING ADVANCED ML PIPELINE ENGINE [{MODEL_TYPE.upper()}] ===")
    
    # Load historical datasets
    train_df = pd.read_csv("data/train_matrix.csv", index_col="Date", parse_dates=True)
    test_df = pd.read_csv("data/test_matrix.csv", index_col="Date", parse_dates=True)
    
    # ------------------------------------------------------------------
    # DYNAMIC TARGET CONFIGURATION LAYER
    # ------------------------------------------------------------------
    if COLLAPSE_TO_BINARY:
        print("💡 ARCHITECTURE CONFIG: Collapsing targets to BINARY [Skip (0) vs Buy (1)]")
        y_train = train_df['target'].apply(lambda x: 1 if x == 1 else 0)
        y_test = test_df['target'].apply(lambda x: 1 if x == 1 else 0)
        eval_labels = [0, 1]
        eval_names = ['Skip (0)', 'Buy (1)']
        report_title = "Binary Optimization"
    else:
        print("⚖️ ARCHITECTURE CONFIG: Retaining BASELINE 3-CLASS [Sell (-1), Hold (0), Buy (1)]")
        y_train = train_df['target']
        y_test = test_df['target']
        eval_labels = [-1, 0, 1]
        eval_names = ['Sell (-1)', 'Hold (0)', 'Buy (1)']
        report_title = "Multi-Class Baseline"
    
    # DROPPING CAPM_BETA HERE to strip away the systemic bubble panic bias
    cols_to_drop = ['target', 'asset_close', 'spy_close', 'vix_close', 'asset_volume', 'capm_beta']
    cols_to_drop = [col for col in cols_to_drop if col in train_df.columns]
    
    X_train_raw = train_df.drop(columns=cols_to_drop)
    X_test_raw = test_df.drop(columns=cols_to_drop)
    
    # Inject advanced features and velocities
    X_train = inject_advanced_features(X_train_raw)
    X_test = inject_advanced_features(X_test_raw)
    
    # ------------------------------------------------------------------
    # PIPELINE ENGINE SELECTION
    # ------------------------------------------------------------------
    if MODEL_TYPE == 'logistic_regression':
        stock_pipeline = Pipeline(steps=[
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('standard_scaler', StandardScaler()),
            ('rfe', RFE(
                estimator=LogisticRegression(C=1.0, max_iter=500, solver='lbfgs', tol=1e-3, class_weight='balanced'),
                n_features_to_select=8,
                step=0.1
            )),
            ('classifier', LogisticRegression(max_iter=500, solver='lbfgs', class_weight='balanced'))
        ])
        param_grid = {'classifier__C': [0.1, 1, 10, 100]}
        
    elif MODEL_TYPE == 'random_forest':
        # NOTE: Random Forests do not need scaling, 
        # but keeping them inside the pipeline keeps architecture uniform!
        stock_pipeline = Pipeline(steps=[
            ('standard_scaler', StandardScaler()),
            ('rfe', RFE(
                estimator=RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced_subsample'),
                n_features_to_select=8,
                step=0.1
            )),
            ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced_subsample'))
        ])
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [5, 10, None]
        }
    else:
        raise ValueError("Invalid MODEL_TYPE chosen.")

    # Hyperparameter Optimization Setup
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_stock_search = GridSearchCV(
        estimator=stock_pipeline,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring='f1_macro', 
        n_jobs=-1
    )
    
    print("Executing optimization sweep across the pipeline matrix...")
    grid_stock_search.fit(X_train, y_train)
    
    best_pipeline = grid_stock_search.best_estimator_
    print(f"\nOptimal Pipeline Parameters Discovered: {grid_stock_search.best_params_}")
    
    # Evaluate performance
    y_pred = best_pipeline.predict(X_test)
    
    print(f"\n=== PIPELINE ENGINE PERFORMANCE METRICS ({report_title.upper()}) ===")
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    print("\nDetailed Financial Classification Report:")
    print(classification_report(y_test, y_pred, labels=eval_labels, target_names=eval_names, zero_division=0))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=eval_labels))
    
    # Trace features kept by RFE step
    rfe_step = best_pipeline.named_steps['rfe']
    selected_features = [X_train.columns[i] for i in range(len(X_train.columns)) if rfe_step.support_[i]]
    
    print("\nTop Retained Features Selected by RFE:")
    for f in selected_features[:8]:
        print(f" └── Selected: {f}")

if __name__ == "__main__":
    train_optimized_pipeline()