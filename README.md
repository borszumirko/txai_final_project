# Final project for Trustworthy & Explainable AI



This project investigates the robustness of saliency maps (CAM variants) when applied to original images, adversarial examples, and noise-perturbed versions of these. The idea is to detect adversarial inputs by measuring the change in heatmaps and label predictions when further perturbations (e.g., Gaussian noise) are added.

---

## Structure

- `main.py` – Run the experiment based on the config file.
- `config.json` – Configuration file with all tunable parameters.
- `results/` – Output files containing reuslts from the experiment
- `images/` – Figures and visualizations (ROC curves and histograms).
- `Small-ImageNet-Validation-Dataset-1000-Classes/` – Subset of ImageNet used for experiments.
- `experiment/` – Code implementing the main experiment
- `plotting/` – Code used to create plots 
- `utils/` – CAM variants, distance metrics and and FGSM, PGD implementations 

---

## Configuration File (`config.json`)

The config file controls experiment settings. Here's a breakdown of the parameters:

### **Adversarial Settings**

- **`adversarial_attack`**: Type of adversarial attack to use (`FGSM` or `PGD`).
- **`epsilon`**: Maximum perturbation for FGSM.
- **`PGD`**:
  - **`alpha`**: Step size per iteration.
  - **`delta`**: Maximum allowed perturbation.
  - **`iterations`**: Number of iterations for the PGD attack.
- **`delta`**, **`max_delta`**: Parameters for FGSM.


---

### **Saliency Map Settings**

- **`saliency_map`**: List of CAM methods to apply. Options include:
  - `gradcam`
  - `gradcam++`
  - `eigen`
  - `layer`

---

### **Metric Parameters**

- **`distance_metrics`**: Metrics used to compare saliency maps:
  - `cosine`: Cosine similarity.
  - `squared`: L2 squared distance.
  - `absolute`: L1 distance.
  - `iou_50`, `iou_75`, `iou_90`: Intersection-over-Union after binarizing with thresholds 0.5, 0.75, 0.90.

---

### **Noise Parameters**

- **`num_noise_vectors`**: Number of noisy samples generated per image.
- **`noise_level`**: Scale noise sampled from normal distribution

---

### **Model & Dataset**

- **`model`**: Pretrained model used for classification. Currently::
  - `"resnet50"` (via `torchvision.models`)
- **`dataset_path`**: Path to the validation dataset subset.
- **`imagenet_labels`**: Path to the ImageNet class index-to-label JSON file.

---

### **Preprocessing Parameters**

- **`center_crop`**: Final crop size for input images (typically 224 for ResNet).
- **`mean`**, **`std`**: Normalization parameters (ImageNet standard).

---

### **Result Paths**

- **`results`**: File names where distance/similarity values will be saved.
  - e.g., `iou_50.txt`, `cosine.txt`, etc.

---

## Running the Experiment

```bash
python main.py
