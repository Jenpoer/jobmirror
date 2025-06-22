"""
Compare the ground truth and predictions
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import roc_auc_score, fbeta_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import seaborn as sns
import json

def get_model_metrics(pred_df, gt_df, date_str, metrics_dir, spark):
    predictions_pd = pred_df.toPandas()
    ground_truth_pd= gt_df.toPandas()

    print("Getting Metrics")
    # 1) Accuracy
    accuracy = accuracy_score(ground_truth_pd["label"], predictions_pd["predicted_label"])

    # 2) ROC_AUC 
    roc_auc = roc_auc_score(ground_truth_pd["label"], predictions_pd["predicted_label"])

    # 3) F-Score 
    f1 = fbeta_score(ground_truth_pd["label"], predictions_pd["predicted_label"], beta=1.0)

    # 4) Confusion Matrix
    cm = confusion_matrix(ground_truth_pd["label"], predictions_pd["predicted_label"])

    print("Saving Metrics")
    metrics_dict = {
        "date" : date_str,
        "accuracy" : accuracy,
        "roc_auc" : roc_auc,
        "f1" : f1
    }
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    cm_save_path = os.path.join(metrics_dir, f"{date_str}_confusion_matrix.png")
    plt.savefig(cm_save_path, bbox_inches='tight')
    plt.close()  # Don't forget to close in script files to avoid memory leaks

    # Save results
    results_json_file_dir = os.path.join(metrics_dir, f"{date_str}_metrics.json")
    with open(results_json_file_dir, 'w') as results_json_file:
        json.dump(metrics_dict, results_json_file)

        