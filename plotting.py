from captum.attr import visualization as viz
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
import PIL
import os
import json
from utils.advesarial import create_FGSM_example, create_PGD_example
from utils.gradcam import create_gradcam, create_gradcam_pp, create_eigencam, create_lrp
from utils.distance import normalize_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from tutorial code
def invert_transform(img_tensor):
    m = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1)
    s = torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1)
    img_numpy = (s * img_tensor + m) * 255
    img_numpy = img_numpy.permute(1,2,0)
    img_numpy = img_numpy.detach().numpy()
    return img_numpy.astype("uint8")

print(f"Using device: {device}")

with open("config.json", "r") as f:
    config = json.load(f)

model = torch.hub.load('pytorch/vision:v0.6.0', config["model"], weights="ResNet50_Weights.IMAGENET1K_V1").to(device)

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

classindex = 18
file = 0
directory = f"{config['dataset_path']}/{classindex}"
files = sorted(os.listdir(directory))

img = load_img(f'{directory}/{files[file]}')

if not img.mode == 'RGB':
    img = img.convert('RGB')

img_tensor = img_preprocessing(img).to(device)

adv_example, orig_copy = create_PGD_example(model, img_tensor, classindex, device, config["PGD"]["alpha"],config["PGD"]["delta"], config["PGD"]["iterations"] )
# adv_example, orig_copy = perform_pgd_attack(model, img_tensor, classindex, device)
# adv_example, orig_copy = create_FGSM_example(model, img_tensor, classindex, config["delta"], device)



pred_orig = torch.argmax(model(img_tensor.unsqueeze(0).to(device)))
pred_adv = torch.argmax(model(adv_example.unsqueeze(0).to(device)))


gradcam_orig, gradcam_adv = create_gradcam(model, img_tensor, adv_example, device)
gradcampp_orig, gardcampp_adv = create_gradcam_pp(model, img_tensor, adv_example, device)
eigencam_orig, eigencam_adv = create_eigencam(model, img_tensor, adv_example, device)
lrp_orig, lrp_adv = create_lrp(model, img_tensor, adv_example, device, pred_orig, pred_adv)


import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T

def display_saliency_visuals(original_img_tensor, adversarial_img_tensor, saliency_maps, titles):
    """
    Displays the original image, adversarial example, and corresponding saliency maps.
    
    Parameters:
    - original_img_tensor: The tensor of the original image.
    - adversarial_img_tensor: The tensor of the adversarial image.
    - saliency_maps: A list of tuples, where each tuple contains the saliency map for both the original and adversarial images.
    - lrp_maps: List of tuples containing LRP saliency maps for the original and adversarial images.
    - titles: A list of titles for the subplots.
    """
    
    original_img = invert_transform(original_img_tensor)
    adversarial_img = invert_transform(adversarial_img_tensor)

    # Create a figure with subplots
    fig, axs = plt.subplots(len(saliency_maps), 4, figsize=(20, 5 * (len(saliency_maps))))

    for i, (saliency_map_original, saliency_map_adv) in enumerate(saliency_maps):
        saliency_map_original = saliency_map_original.detach()
        saliency_map_adv = saliency_map_adv.detach()
        
        # Plot the original image
        axs[i, 0].imshow(original_img)
        axs[i, 0].axis('off')
        axs[i, 0].set_title(f"Original Image, pred: {pred_orig}")
        
        # Plot the adversarial image
        axs[i, 1].imshow(adversarial_img)
        axs[i, 1].axis('off')
        axs[i, 1].set_title(f"Adversarial Image, pred:{pred_adv}")

        # Overlay the saliency map on the original image
        axs[i, 2].imshow(original_img)
        axs[i, 2].imshow(saliency_map_original, cmap='jet', alpha=0.5)
        axs[i, 2].axis('off')
        axs[i, 2].set_title(f"{titles[i]} (Original)")

        # Overlay the saliency map on the adversarial image
        axs[i, 3].imshow(adversarial_img)
        axs[i, 3].imshow(saliency_map_adv, cmap='jet', alpha=0.5)
        axs[i, 3].axis('off')
        axs[i, 3].set_title(f"{titles[i]} (Adversarial)")

    plt.tight_layout()
    plt.show()


# display_saliency_visuals(
#     img_tensor, adv_example,
#     saliency_maps=[(gradcam_orig, gradcam_adv), (gradcampp_orig, gardcampp_adv), (eigencam_orig, eigencam_adv)],
#     titles=["GradCAM", "GradCAM++", "EigenCam"]
# )

# from https://captum.ai/tutorials/TorchVision_Interpret
_ = viz.visualize_image_attr_multiple(np.transpose(lrp_orig.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      invert_transform(img_tensor),
                                      ["original_image", "heat_map"],
                                      ["all", "positive"],
                                      show_colorbar=True,
                                      outlier_perc=2)

_ = viz.visualize_image_attr_multiple(np.transpose(lrp_adv.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      invert_transform(adv_example),
                                      ["original_image", "heat_map"],
                                      ["all", "positive"],
                                      show_colorbar=True,
                                      outlier_perc=2)
