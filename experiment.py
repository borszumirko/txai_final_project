import torch
import torchvision.transforms as T
import json
import PIL
import os
import numpy as np
from tqdm import tqdm
from utils.distance import calculate_distance
from utils.gradcam import create_gradcam, create_gradcam_pp, calculate_saliency
from utils.advesarial import create_FGSM_example, create_PGD_example


def run_experiment(config_file, device):
    
    with open(config_file, "r") as f:
        config = json.load(f)

    model = torch.hub.load('pytorch/vision:v0.6.0', config["model"], weights="ResNet50_Weights.IMAGENET1K_V1").to(device)
    # Prerocess for resnet50
    # From tutorial code
    img_preprocessing = T.Compose([
        T.CenterCrop(config["preprocessing"]["center_crop"]),
        T.ToTensor(),
        T.Normalize(mean=config["preprocessing"]["mean"], std=config["preprocessing"]["std"])
    ])

    def invert_transform(img_tensor):
        m = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1)
        s = torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1)
        img_numpy = (s * img_tensor + m) * 255
        img_numpy = img_numpy.permute(1,2,0)
        img_numpy = img_numpy.detach().numpy()
        return img_numpy.astype("uint8")


    # Load and preprocess image
    def load_img(path):
        image = PIL.Image.open(path)
        return image

    # Load ImageNet labels
    with open(config["imagenet_labels"]) as f:
        imagenet_labels = json.load(f)


    distances = {
        heatmap_type: {
            metric: {"orig": [], "adv": []}
            for metric in config["distance_metrics"]
        }
        for heatmap_type in config["saliency_map"]
    }

    model_mistake = 0
    adversarial_mistake = 0

    for classindex in tqdm(range(1000), desc="Class"):
        for file in tqdm(range(5), desc="file", leave=False):
            directory = f"{config['dataset_path']}/{classindex}"
            files = sorted(os.listdir(directory))

            img = load_img(f'{directory}/{files[file]}')

            if not img.mode == 'RGB':
                img = img.convert('RGB')

            img_tensor = img_preprocessing(img).to(device)

            model.eval()

            if config["adversarial_attack"] == "FGSM":
                adversarial_example, img_tensor_copy = create_FGSM_example(model, img_tensor, classindex, delta=config["delta"], device=device)
            elif config["adversarial_attack"] == "PGD":
                adversarial_example, img_tensor_copy = create_PGD_example(model, img_tensor, classindex, device, config["PGD"]["alpha"],config["PGD"]["delta"], config["PGD"]["iterations"] )

            pred_orig =  model(img_tensor.unsqueeze(0)).argmax()
            idx_orig = pred_orig.item()
            
            # Only consider images that are correctly classified
            if idx_orig != classindex:
                model_mistake +=1
                continue

            pred_adv = model(adversarial_example.unsqueeze(0)).argmax()
            idx_adv = pred_adv.item()

            # Only proceed if the adversarial example is misclassified
            # Increase delta until the adversarial example is misclassified
            delta = config["delta"]
            while classindex == idx_adv and delta < config["max_delta"]:
                delta += 0.01
                adversarial_example = img_tensor_copy + delta * torch.sign(img_tensor.grad)
                idx_adv = model(img_tensor.unsqueeze(0)).argmax().item()
                label_adv = imagenet_labels[idx_adv]
            
            if idx_orig == idx_adv:
                adversarial_mistake +=1
                continue
            

            saliency_maps_no_noise = {saliency: {"orig": None, "adv": None} for saliency in config["saliency_map"]}

            for heatmap_type in config["saliency_map"]:
                gradcam_orig, gradcam_adv = calculate_saliency(model, img_tensor, adversarial_example, device, pred_orig, pred_adv, heatmap_type=heatmap_type)
                saliency_maps_no_noise[heatmap_type]["orig"] = gradcam_orig
                saliency_maps_no_noise[heatmap_type]["adv"] = gradcam_adv
            


            num_noises = config["num_noise_vectors"]

            noise_vectors = [torch.randn_like(img_tensor, device=device) * config["noise_level"] for _ in range(num_noises)]

            differences = {
                heatmap_type: {
                    metric: {"orig": [], "adv": []}
                    for metric in config["distance_metrics"]
                }
                for heatmap_type in config["saliency_map"]
            }
            # Calculate differences for each noise vector and append to differences dictionary 
            for noise in noise_vectors:

                noisy_orig = img_tensor + noise
                noisy_adv = adversarial_example + noise
                

                for heatmap_type in config["saliency_map"]:
                    gradcam_noisy_orig, gradcam_noisy_adv = calculate_saliency(model, noisy_orig, noisy_adv, device, pred_orig, pred_adv, heatmap_type)
                    for distance_metric in config['distance_metrics']:
                        
                        dist_orig = calculate_distance(saliency_maps_no_noise[heatmap_type]["orig"], gradcam_noisy_orig, distance_metric)
                        dist_adv = calculate_distance(saliency_maps_no_noise[heatmap_type]["adv"], gradcam_noisy_adv, distance_metric)

                        differences[heatmap_type][distance_metric]["orig"].append(dist_orig)
                        differences[heatmap_type][distance_metric]["adv"].append(dist_adv)

            # Append mean results to distances dictionary
            for heatmap_type in config["saliency_map"]:
                for distance_metric, values in differences[heatmap_type].items():

                    values["orig"] = [t.cpu().detach() for t in values["orig"]]
                    values["adv"] = [t.cpu().detach() for t in values["adv"]]

                    values["orig"] = np.array(values["orig"])
                    values["adv"] = np.array(values["adv"])
                    
                    mean_orig = np.mean(values["orig"])
                    mean_adv = np.mean(values["adv"])

                    distances[heatmap_type][distance_metric]["orig"].append(mean_orig)
                    distances[heatmap_type][distance_metric]["adv"].append(mean_adv)

                    # Write results to file
                    with open(f"results/{config['adversarial_attack']}_{heatmap_type}_{config['results'][distance_metric]}", "a") as f:
                        f.write(f"{classindex},{files[file]},{mean_orig},{mean_adv}\n")

    '''with open(f"results/mistakes.txt", "a") as f:
        f.write(f"Model making a mistake on original image:{model_mistake}\n")
        f.write(f"Model not fooled by adversarial image:{adversarial_mistake}")'''

    # Print results
    for distance_metric, values in distances.items():
        mean_orig = np.mean(values["orig"])
        mean_adv = np.mean(values["adv"])
        print(f"Distance metric: {distance_metric}")
        print(f"Mean difference for original image: {mean_orig}")
        print(f"Mean difference for adversarial example: {mean_adv}")
        print("\n")


def label_experiment(device, config_file):
    with open(config_file, "r") as f:
        config = json.load(f)

    model = torch.hub.load('pytorch/vision:v0.6.0', config["model"], weights="ResNet50_Weights.IMAGENET1K_V1").to(device)
    # Prerocess for resnet50
    # From tutorial code
    img_preprocessing = T.Compose([
        T.CenterCrop(config["preprocessing"]["center_crop"]),
        T.ToTensor(),
        T.Normalize(mean=config["preprocessing"]["mean"], std=config["preprocessing"]["std"])
    ])

    # Load and preprocess image
    def load_img(path):
        image = PIL.Image.open(path)
        return image

    model_mistake = 0
    FGSM_mistake = 0
    PGD_mistake = 0

    label_changes = {
                    "orig_FGSM": [],
                    "orig_PGD": [],
                    "FGSM": [],
                    "PGD": [],
                    }

    for classindex in tqdm(range(1000), desc="Class"):
        for file in tqdm(range(5), desc="file", leave=False):
            directory = f"{config['dataset_path']}/{classindex}"
            files = sorted(os.listdir(directory))

            img = load_img(f'{directory}/{files[file]}')

            if not img.mode == 'RGB':
                img = img.convert('RGB')

            img_tensor = img_preprocessing(img).to(device)

            model.eval()

            FGSM_example, img_tensor_copy = create_FGSM_example(model, img_tensor, classindex, delta=config["delta"], device=device)
            
            PGD_example, img_tensor_copy = create_PGD_example(model, img_tensor, classindex, device, config["PGD"]["alpha"],config["PGD"]["delta"], config["PGD"]["iterations"] )

            pred_orig =  model(img_tensor.unsqueeze(0)).argmax()
            idx_orig = pred_orig.item()
            
            if idx_orig != classindex:
                model_mistake +=1
                continue

            pred_FGSM = model(FGSM_example.unsqueeze(0)).argmax()
            idx_FGSM = pred_FGSM.item()
            pred_PGD = model(PGD_example.unsqueeze(0)).argmax()
            idx_PGD = pred_PGD.item()
            
            delta = config["delta"]
            while classindex == idx_FGSM and delta < config["max_delta"]:
                delta += 0.01
                FGSM_example = img_tensor_copy + delta * torch.sign(img_tensor.grad)
                idx_FGSM = model(img_tensor.unsqueeze(0)).argmax().item()
            
            FGSM_successful = True
            PGD_successful = True

            if idx_orig == idx_FGSM:
                FGSM_mistake +=1
                FGSM_successful = False

            if idx_orig == idx_PGD:
                PGD_mistake +=1
                PGD_successful = False
            
            if not (FGSM_successful and PGD_successful):
                continue

            num_noises = config["num_noise_vectors"]

            noise_vectors = [torch.randn_like(img_tensor, device=device) * config["noise_level"] for _ in range(num_noises)]

            label_change_rate = 0.0
            label_change_rate_FGSM = 0.0
            label_change_rate_PGD = 0.0

            for noise in noise_vectors:

                noisy_orig = (img_tensor + noise).to(device)
                noisy_FGSM = (FGSM_example + noise).to(device)
                noisy_PGD = (PGD_example + noise).to(device)

                idx_orig_noisy = model(noisy_orig.unsqueeze(0)).argmax().item()
                
                if idx_orig_noisy != idx_orig:
                    label_change_rate +=1

                if FGSM_successful:
                    idx_FGSM_noisy = model(noisy_FGSM.unsqueeze(0)).argmax().item()
                    if idx_FGSM_noisy != idx_FGSM:
                        label_change_rate_FGSM += 1
                
                if PGD_successful:
                    idx_PGD_noisy = model(noisy_PGD.unsqueeze(0)).argmax().item()
                    if idx_PGD_noisy != idx_PGD:
                        label_change_rate_PGD += 1
            
            if FGSM_successful:
                with open("results/FGSM_label_change.txt", "a") as f:
                    f.write(f"{label_change_rate},{label_change_rate_FGSM / num_noises}")

            if PGD_successful:
                with open("results/PGD_label_change.txt", "a") as f:
                    f.write(f"{label_change_rate},{label_change_rate_PGD / num_noises}")






                

                
                
            




            