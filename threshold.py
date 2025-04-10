import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics import roc_curve, f1_score
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
best_overall_f1 = 0
best_overall_threshold = 0
best_metric = ""
for adversarial in adversarials:
    for saliency in saliency_methods:
        for distance_metric in config["distance_metrics"]:
            orig, adv = extract_last_two_numbers(f"results/{adversarial}_{saliency}_{distance_metric}.txt")
            if distance_metric  not in ["squared", "absolute"]:
                    orig = [-x for x in orig]
                    adv = [-x for x in adv]

            y_true = np.array([0]*len(orig) + [1]*len(adv))
            y_scores = np.concatenate([orig, adv])
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            best_f1 = 0
            best_threshold = 0
            for t in thresholds:
                if np.isinf(t):
                    continue
                preds = (y_scores >= t).astype(int)
                f1 = f1_score(y_true, preds)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = t
            results.append({
                "Adversarial Method": adversarial,
                "Saliency Method": saliency,
                "Distance Metric": distance_metric,
                "Best F1 Score": best_f1,
                "Best Threshold": best_threshold
            })
            if best_f1 > best_overall_f1:
                best_overall_f1 = best_f1
                best_metric = distance_metric
                best_overall_threshold = best_threshold
        best_results.append({
                "Adversarial Method": adversarial,
                "Saliency Method": saliency,
                "Best Distance Metric": best_metric,
                "Best F1 Score": best_overall_f1,
                "Best Threshold": best_overall_threshold
            })

#print(f"best metric: {best_metric}, f1: {best_overall_f1}, threshold: {best_overall_threshold}")
output_file = "threshold_results.csv"
with open(output_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["Adversarial Method", "Saliency Method", "Distance Metric", "Best F1 Score", "Best Threshold"])
    writer.writeheader()
    writer.writerows(results)
output_file_best = "best_metrics_threshold.csv"
with open(output_file_best, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["Adversarial Method", "Saliency Method", "Best Distance Metric", "Best F1 Score", "Best Threshold"])
    writer.writeheader()
    writer.writerows(best_results)

'''
plt.figure()
plt.hist(orig, bins=60, alpha=0.5, label="original", color="blue")
plt.hist(adv, bins=60, alpha=0.5, label="Adversarial", color="red")
plt.xlabel("Distance Value")
plt.ylabel("Frequency")
plt.title(f"saliency map: saliency, distance: metric")
plt.legend()
plt.savefig("results/thresholding.png")
plt.close()

epsilon = 1e-8
orig = np.array(orig, dtype=np.float32)
adv = np.array(adv, dtype=np.float32)
orig = orig + epsilon
adv = adv + epsilon
orig /= np.sum(orig)
adv /= np.sum(adv)

kl = entropy(pk=orig, qk=adv)
wd = wasserstein_distance(orig, adv)
print(f"KL: {kl}, WD: {wd}")


print(f"orig length: {len(orig)}, adv: {len(adv)}")
y_true = np.array([0]*len(orig) + [1]*len(adv))
y_scores = np.concatenate([orig, adv])
print(f"y_true length: {len(y_true)}, y_scores: {len(y_scores)}")
fpr, tpr, thresholds = roc_curve(y_true, y_scores)


print(f"{fpr}\n, {tpr}\n, {thresholds}\n")

best_f1 = 0
best_threshold = 0

for t in thresholds:
    preds = (y_scores >= t).astype(int)
    f1 = f1_score(y_true, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"Best threshold by F1 score: {best_threshold:.4f}")
print(f"Best F1 score: {best_f1:.4f}")'''