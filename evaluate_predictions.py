import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

def load_actuals(filepath='data/studentInfo.csv'):
    """Loads the actual outcomes from the student info dataset."""
    try:
        df = pd.read_csv(filepath)
        # Match the target definition in assign_scores.py
        # Pass and Distinction are 1 (Pass), else 0 (Fail)
        df['actual_target'] = df['final_result'].apply(lambda x: 1 if x in ['Pass', 'Distinction'] else 0)
        return df[['id_student', 'code_module', 'code_presentation', 'actual_target']]
    except FileNotFoundError:
        print(f"Error: Could not find actuals data at {filepath}")
        return pd.DataFrame()

def load_predictions(filepath):
    """Loads predictions from a JSON file and formats them as a DataFrame."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        rows = []
        for item in data:
            rows.append({
                'id_student': item['student_id'],
                'code_module': item['course_module'],
                'code_presentation': item['presentation'],
                'predicted_class': 1 if item['prediction']['predicted_class'] == 'Pass' else 0,
                'prob_pass': item['prediction']['probability_pass']
            })
        return pd.DataFrame(rows)
    except FileNotFoundError:
        print(f"Warning: Prediction file not found: {filepath}")
        return pd.DataFrame()

def evaluate():
    print("Loading actual targets...")
    actuals_df = load_actuals()
    if actuals_df.empty:
        return

    print("Loading prediction files...")
    rf_df = load_predictions('rf_predictions.json')
    lstm_df = load_predictions('lstm_predictions.json')
    
    if rf_df.empty and lstm_df.empty:
        print("No prediction files found. Please run assign_scores.py first to generate the predictions.")
        return

    # Merge predictions with actual targets
    models = {}
    if not rf_df.empty:
        models['Random Forest'] = pd.merge(rf_df, actuals_df, on=['id_student', 'code_module', 'code_presentation'], how='inner')
    if not lstm_df.empty:
        models['LSTM'] = pd.merge(lstm_df, actuals_df, on=['id_student', 'code_module', 'code_presentation'], how='inner')

    metrics = {}
    output_pdf = 'evaluation_results.pdf'
    
    print(f"Evaluating models and generating plots into {output_pdf}...")
    with PdfPages(output_pdf) as pdf:
        
        # 1. ROC Curves Plot
        plt.figure(figsize=(8, 6))
        for name, df in models.items():
            y_true = df['actual_target']
            y_prob = df['prob_pass']
            y_pred = df['predicted_class']
            
            # Calculate ROC and AUC
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
            
            # Store general metrics for the bar chart
            metrics[name] = {
                'Accuracy': accuracy_score(y_true, y_pred),
                'Precision': precision_score(y_true, y_pred, zero_division=0),
                'Recall': recall_score(y_true, y_pred, zero_division=0),
                'F1 Score': f1_score(y_true, y_pred, zero_division=0)
            }
            
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        pdf.savefig()
        plt.close()
        
        # 2. Confusion Matrices Plot
        if models:
            fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
            # Make axes iterable if there's only one model
            if len(models) == 1:
                axes = [axes]
                
            for ax, (name, df) in zip(axes, models.items()):
                y_true = df['actual_target']
                y_pred = df['predicted_class']
                cm = confusion_matrix(y_true, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fail', 'Pass'])
                disp.plot(ax=ax, cmap='Blues', colorbar=False)
                ax.set_title(f'{name} Confusion Matrix')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # 3. Performance Metrics Bar Chart
        if metrics:
            metrics_df = pd.DataFrame(metrics).T
            ax = metrics_df.plot(kind='bar', figsize=(10, 6), rot=0)
            plt.title('Model Performance Metrics Comparison')
            plt.ylabel('Score')
            plt.ylim(0, 1.15) # Leave room for labels
            plt.grid(axis='y', alpha=0.3)
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)
            
            # Add numeric labels to the bars
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.2f}", 
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', 
                            xytext=(0, 5), textcoords='offset points', 
                            fontsize=9)
                            
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    print("Evaluation complete!")
    print(f"Visualizations saved to '{output_pdf}'.")
    
    if metrics:
        print("\nMetrics Summary:")
        metrics_summary_df = pd.DataFrame(metrics).T
        print(metrics_summary_df.round(3).to_string())

if __name__ == "__main__":
    evaluate()
