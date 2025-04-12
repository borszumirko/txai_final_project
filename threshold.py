import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics import roc_curve, f1_score
from sklearn.model_selection import train_test_split
import csv

with open("config.json", "r") as f:
    config = json.load(f)

def extract_last_two_numbers(file_path):
    orig = []
    adv = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                last_two = parts[-2:]  # Get the last two elements
                orig.append(float(last_two[0]))
                adv.append(float(last_two[1]))
    return orig, adv


adversarials = ["FGSM", "PGD"]
saliency_methods = ["gradcam", "gradcam++", "eigen", "layer"]
results = []
best_results = []


for adversarial in adversarials:
    for saliency in saliency_methods:
        best_test_f1 = 0
        best_test_threshold = 0
        best_metric = ""
        best_validation_f1 = 0

        for distance_metric in config["distance_metrics"]:
            #Read the distance from the txt file
            orig, adv = extract_last_two_numbers(f"results/{adversarial}_{saliency}_{distance_metric}.txt")
            #Make sure that all metrics are aligned
            if distance_metric  not in ["squared", "absolute"]:
                    orig = [-x for x in orig]
                    adv = [-x for x in adv]
            
            #Make the y and x lists
            y = np.array([0]*len(orig) + [1]*len(adv))
            x = np.concatenate([orig, adv])
            #Test and validation split
            x_test, x_val, y_test, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
            
            #ROC values and thresholds
            fpr, tpr, thresholds = roc_curve(y_test, x_test)

            best_f1 = 0
            best_threshold = 0

            #Get the best threshold for the metric with respect to f1 score on test set
            for t in thresholds:
                if np.isinf(t):
                    continue
                preds = (x_test >= t).astype(int)
                f1 = f1_score(y_test, preds)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = t
            
            #Calculate f1 score on validation set with the same threshold
            val_preds = (x_val >= best_threshold).astype(int)
            f1_val = f1_score(y_val, val_preds)

            #Check if this metric has the best validation f1 score
            if f1_val > best_validation_f1:
                best_validation_f1 = f1_val
                best_metric = distance_metric
                best_test_threshold = best_threshold
                best_test_f1 = best_f1

        results.append({
                "Adversarial Method": adversarial,
                "Saliency Method": saliency,
                "Best Distance Metric": best_metric,
                "Best Test F1 Score": best_test_f1,
                "Best Threshold": best_test_threshold,
                "Validation F1 Score": best_validation_f1
            })

output_file = "validation_threshold_results.csv"
with open(output_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["Adversarial Method", "Saliency Method", "Best Distance Metric", "Best Test F1 Score", "Best Threshold", "Validation F1 Score"])
    writer.writeheader()
    writer.writerows(results)

