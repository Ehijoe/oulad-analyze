import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- 1. Preprocessing Function ---
def preprocess_oulad_data():
    """Loads, merges, and cleans OULAD data for all courses."""
    # Load raw data
    s_info = pd.read_csv('data/studentInfo.csv')
    s_reg = pd.read_csv('data/studentRegistration.csv')
    s_assess = pd.read_csv('data/studentAssessment.csv')
    assess = pd.read_csv('data/assessments.csv')
    s_vle = pd.read_csv('data/studentVle.csv')

    # Merge demographics and registration
    df = pd.merge(s_info, s_reg, on=['id_student', 'code_module', 'code_presentation'])
    
    # Calculate assessment features
    assess_data = pd.merge(s_assess, assess, on='id_assessment')
    assess_data['score'] = assess_data['score'].fillna(0)
    student_scores = assess_data.groupby(['id_student', 'code_module', 'code_presentation']).agg(
        avg_score=('score', 'mean'),
        failed_count=('score', lambda x: (x < 40).sum())
    ).reset_index()
    
    # Merge VLE interactions
    vle_data = s_vle.groupby(['id_student', 'code_module', 'code_presentation'])['sum_click'].sum().reset_index()
    
    # Final merge
    df = pd.merge(df, student_scores, on=['id_student', 'code_module', 'code_presentation'], how='left')
    df = pd.merge(df, vle_data, on=['id_student', 'code_module', 'code_presentation'], how='left')
    
    # Clean and Target
    df = df.fillna(0)
    df['target'] = df['final_result'].apply(lambda x: 1 if x in ['Pass', 'Distinction'] else 0)
    return df

# --- 2. Random Forest Function ---
def run_random_forest(df):
    results = []
    # Drop non-feature columns
    features_df = df.drop(columns=['target', 'id_student', 'code_module', 'code_presentation', 'final_result'])
    
    # ONE-HOT ENCODING: This converts 'M'/'F' and other strings into numeric columns
    features_df = pd.get_dummies(features_df) 
    
    for module in df['code_module'].unique():
        # Filter rows for the current module
        mask = df['code_module'] == module
        X = features_df[mask]
        y = df.loc[mask, 'target']
        
        rf = RandomForestClassifier(n_estimators=100) # [cite: 224]
        rf.fit(X, y)
        probs = rf.predict_proba(X)[:, 1]
        
        for idx, (original_idx, row) in enumerate(df[mask].iterrows()):
            p_pass = float(probs[idx])
            results.append({
                "student_id": int(row['id_student']),
                "course_module": module,
                "presentation": row['code_presentation'],
                "diagnostics": {
                    "vle_clicks": int(row['sum_click']),
                    "avg_assessment_score": round(float(row['avg_score']), 1),
                    "prior_fails": int(row['failed_count'])
                },
                "prediction": {
                    "predicted_class": "Pass" if p_pass >= 0.5 else "Fail",
                    "probability_pass": round(p_pass, 2),
                    "probability_fail": round(1.0 - p_pass, 2)
                },
            })
    with open('rf_predictions.json', 'w', encoding="utf-8") as f:
        json.dump(results, f, indent=2)

# --- 3. LSTM Function ---
def run_lstm(df):
    results = []
    # 1. Pre-encode all features as numeric
    features_df = pd.get_dummies(df.drop(columns=['target', 'id_student', 'code_module', 'code_presentation', 'final_result']))
    # Ensure all columns are numeric
    features_df = features_df.astype(float) 

    for module in df['code_module'].unique():
        mask = df['code_module'] == module
        X = features_df[mask]
        y = df.loc[mask, 'target']
        
        n_steps, n_features = 1, X.shape[1]
        X_seq = np.tile(X.values[:, np.newaxis, :], (1, n_steps, 1)).astype(np.float32)
        
        # Build and train model
        model = Sequential([
            LSTM(300, input_shape=(n_steps, n_features), activation='sigmoid'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam')
        model.fit(X_seq, y, epochs=5, verbose=0)
        
        probs = model.predict(X_seq).flatten()
        
        # Iterate using enumerate to match indices correctly
        for idx, (original_idx, row) in enumerate(df[mask].iterrows()):
            p_pass = float(probs[idx])
            results.append({
                "student_id": int(row['id_student']),
                "course_module": module,
                "presentation": row['code_presentation'],
                "diagnostics": {
                    "vle_clicks": int(row['sum_click']),
                    "avg_assessment_score": round(float(row['avg_score']), 1),
                    "prior_fails": int(row['failed_count'])
                },
                "prediction": {
                    "predicted_class": "Pass" if p_pass >= 0.5 else "Fail",
                    "probability_pass": round(p_pass, 2),
                    "probability_fail": round(1.0 - p_pass, 2)
                },
            })
    with open('lstm_predictions.json', 'w', encoding="utf-8") as f:
        json.dump(results, f, indent=2)

# --- 4. Execution ---
if __name__ == "__main__":
    data = preprocess_oulad_data()
    run_random_forest(data)
    run_lstm(data)
    print("Predictions saved to rf_predictions.json and lstm_predictions.json.")
