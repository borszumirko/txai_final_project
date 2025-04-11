import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import entropy, wasserstein_distance

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

orig, adv = extract_last_two_numbers(f"results/PGD_gradcam++_iou_50.txt")

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