import torch
import torchvision.transforms as T
import json
import PIL
import os
import numpy as np
from tqdm import tqdm
from utils.distance import calculate_distance
from utils.gradcam import create_gradcam
from utils.advesarial import create_FGSM_example

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

with open("config.json", "r") as f:
    config = json.load(f)

model = torch.hub.load('pytorch/vision:v0.6.0', config["model"], pretrained=True).to(device)

# Prerocess for resnet50
img_preprocessing = T.Compose([
    T.CenterCrop(config["preprocessing"]["center_crop"]),
    T.ToTensor(),
    T.Normalize(mean=config["preprocessing"]["mean"], std=config["preprocessing"]["std"])
])


# Load and preprocess image
def load_img(path):
    image = PIL.Image.open(path)
    return image

# Load ImageNet labels
with open(config["imagenet_labels"]) as f:
    imagenet_labels = json.load(f)


distances = {metric: {"orig": [], "adv": []} for metric in config["distance_metrics"]}

for classindex in tqdm(range(1000), desc="Class"):
  for file in range(5):
    directory = f"{config['dataset_path']}/{classindex}"
    files = sorted(os.listdir(directory))

    img = load_img(f'{directory}/{files[file]}')

    if not img.mode == 'RGB':
        img = img.convert('RGB')

    img_tensor = img_preprocessing(img).to(device)

    model.eval()

    if config["adversarial_attack"] == "FGSM":
        adversarial_example, img_tensor_copy = create_FGSM_example(model, img_tensor, classindex, delta=config["delta"], device=device)

    idx_orig = model(img_tensor.unsqueeze(0)).argmax().item()
    
    # Only consider images that are correctly classified
    if idx_orig != classindex:
      continue

    idx_adv = model(adversarial_example.unsqueeze(0)).argmax().item()

    # Only proceed if the adversarial example is misclassified
    # Increase delta until the adversarial example is misclassified
    delta = config["delta"]
    while classindex == idx_adv and delta < config["max_delta"]:
        delta += 0.01
        adversarial_example = img_tensor_copy + delta * torch.sign(img_tensor.grad)
        idx_adv = model(img_tensor.unsqueeze(0)).argmax().item()
        label_adv = imagenet_labels[idx_adv]
    
    if idx_orig == idx_adv:
        continue
    
    gradcam_orig, gradcam_adv = create_gradcam(model, img_tensor, adversarial_example, device)


    num_noises = config["num_noise_vectors"]

    noise_vectors = [torch.randn_like(img_tensor, device=device) * config["noise_level"] for _ in range(num_noises)]

    differences = {metric: {"orig": [], "adv": []} for metric in config["distance_metrics"]}

    # Calculate differences for each noise vector and append to differences dictionary 
    for noise in noise_vectors:

        noisy_orig = img_tensor + noise
        noisy_adv = adversarial_example + noise
        
        gradcam_noisy_orig, gradcam_noisy_adv = create_gradcam(model, noisy_orig, noisy_adv, device)

        for distance_metric in config['distance_metrics']:
            
            dist_orig = calculate_distance(gradcam_orig, gradcam_noisy_orig, distance_metric)
            dist_adv = calculate_distance(gradcam_adv, gradcam_noisy_adv, distance_metric)

            differences[distance_metric]["orig"].append(dist_orig)
            differences[distance_metric]["adv"].append(dist_adv)

    # Append mean results to distances dictionary
    for distance_metric, values in differences.items():

        values["orig"] = [t.cpu() for t in values["orig"]]
        values["adv"] = [t.cpu() for t in values["adv"]]

        values["orig"] = np.array(values["orig"])
        values["adv"] = np.array(values["adv"])
        
        mean_orig = np.mean(values["orig"])
        mean_adv = np.mean(values["adv"])

        distances[distance_metric]["orig"].append(mean_orig)
        distances[distance_metric]["adv"].append(mean_adv)

        # Write results to file
        with open(f"results/{config['results'][distance_metric]}", "a") as f:
            f.write(f"{classindex},{files[file]},{mean_orig},{mean_adv}\n")



# Print results
for distance_metric, values in distances.items():
    mean_orig = np.mean(values["orig"])
    mean_adv = np.mean(values["adv"])
    print(f"Distance metric: {distance_metric}")
    print(f"Mean difference for original image: {mean_orig}")
    print(f"Mean difference for adversarial example: {mean_adv}")
    print("\n")