import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc

def analyze_predictions(real_csv_path, fake_csv_path):
    # 1. Parse REAL videos
    try:
        real_df = pd.read_csv(real_csv_path)
        # All videos in this CSV are actually REAL (Negative class)
        true_negatives = len(real_df[real_df['Prediction'] == 'REAL'])
        false_positives = len(real_df[real_df['Prediction'] == 'FAKE'])
        total_real = len(real_df)
        
        real_df['y_true'] = 0
        real_df['y_score'] = real_df.apply(lambda row: row['Confidence']/100.0 if row['Prediction'] == 'FAKE' else 1.0 - (row['Confidence']/100.0), axis=1)
    except Exception as e:
        print(f"Error reading {real_csv_path}: {e}")
        return

    # 2. Parse FAKE videos 
    try:
        fake_df = pd.read_csv(fake_csv_path)
        # All videos in this CSV are actually FAKE (Positive class)
        true_positives = len(fake_df[fake_df['Prediction'] == 'FAKE'])
        false_negatives = len(fake_df[fake_df['Prediction'] == 'REAL'])
        total_fake = len(fake_df)
        
        fake_df['y_true'] = 1
        fake_df['y_score'] = fake_df.apply(lambda row: row['Confidence']/100.0 if row['Prediction'] == 'FAKE' else 1.0 - (row['Confidence']/100.0), axis=1)
    except Exception as e:
        print(f"Error reading {fake_csv_path}: {e}")
        return

    # 3. Aggregate totals
    total_videos = total_real + total_fake
    total_correct = true_positives + true_negatives
    total_incorrect = false_positives + false_negatives

    # 4. Calculate Metrics
    accuracy = total_correct / total_videos
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / total_fake if total_fake > 0 else 0.0
    specificity = true_negatives / total_real if total_real > 0 else 0.0
    
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    fpr = false_positives / total_real
    fnr = false_negatives / total_fake
    
    # Calculate ROC and AUC
    all_data = pd.concat([real_df, fake_df], ignore_index=True)
    roc_fpr, roc_tpr, _ = roc_curve(all_data['y_true'], all_data['y_score'])
    roc_auc = auc(roc_fpr, roc_tpr)

    # 5. Print Formatted Report
    print("="*50)
    print("DEEPFAKE BATCH PREDICTION ANALYSIS")
    print("="*50)
    print(f"Total Videos Processed: {total_videos:,}")
    print(f"  - Actual REAL Videos: {total_real:,}")
    print(f"  - Actual FAKE Videos: {total_fake:,}")
    print("-" * 50)
    
    print("CONFUSION MATRIX:")
    print(f"  True Positives (Predicted FAKE, Actual FAKE):  {true_positives:,}")
    print(f"  True Negatives (Predicted REAL, Actual REAL):  {true_negatives:,}")
    print(f"  False Positives (Predicted FAKE, Actual REAL): {false_positives:,}  <-- False Alarms")
    print(f"  False Negatives (Predicted REAL, Actual FAKE): {false_negatives:,}  <-- Missed Deepfakes")
    print("-" * 50)
    
    print("CORE METRICS:")
    print(f"  Accuracy:    {accuracy:.2%}  ({total_correct:,}/{total_videos:,})")
    print(f"  Precision:   {precision:.2%}  (When guessing FAKE, it is correct {precision:.2%} of the time)")
    print(f"  Recall(TPR): {recall:.2%}  (Caught {recall:.2%} of all actual FAKEs)")
    print(f"  Specificity: {specificity:.2%}  (Correctly cleared {specificity:.2%} of actual REALs)")
    print(f"  F1-Score:    {f1_score:.2%}")
    print("-" * 50)
    
    print("ERROR RATES:")
    print(f"  False Positive Rate (FPR): {fpr:.2%}")
    print(f"  False Negative Rate (FNR): {fnr:.2%}")
    print("-" * 50)
    print(f"  AUC-ROC:     {roc_auc:.4f}")
    print("="*50)

    # 6. Generate Confusion Matrix Image
    cm = np.array([[true_negatives, false_positives], 
                   [false_negatives, true_positives]])
                   
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted REAL', 'Predicted FAKE'],
                yticklabels=['Actual REAL', 'Actual FAKE'])
    plt.title('Batch Prediction Confusion Matrix')
    plt.tight_layout()
    
    plt.savefig('batch_confusion_matrix.png', dpi=300)
    plt.close()
    
    # 7. Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(roc_fpr, roc_tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('batch_roc_curve.png', dpi=300)
    plt.close()
    
    print("\nVisualizations saved:")
    print(" - batch_confusion_matrix.png")
    print(" - batch_roc_curve.png")

if __name__ == "__main__":
    # Point these to the CSV files in your directory
    REAL_CSV = "real.csv"
    FAKE_CSV = "fake.csv"
    
    analyze_predictions(REAL_CSV, FAKE_CSV)
