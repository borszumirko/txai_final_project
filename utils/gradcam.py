import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus

def create_gradcam(model, img_tensor, adversarial_example, device):
    # Grad-CAM for the original image
    gradcam = GradCAM(model=model, target_layers=[model.layer4[2]])
    gradcam_orig = gradcam(input_tensor=img_tensor.unsqueeze(0))
    gradcam_orig = gradcam_orig[0, :]

    # Grad-CAM for the adversarial example
    gradcam_adv = gradcam(input_tensor=adversarial_example.unsqueeze(0))
    gradcam_adv = gradcam_adv[0, :]

    gradcam_orig = torch.tensor(gradcam_orig).to(device)
    gradcam_adv = torch.tensor(gradcam_adv).to(device)

    return gradcam_orig, gradcam_adv

#GradCAMPlusPlus implementation 
def create_gradcam_pp(model, img_tensor, adversarial_example, device):
    # Grad-CAM-PlusPlus for the original image
    gradcam_pp = GradCAMPlusPlus(model=model, target_layers=[model.layer4[2]])
    gradcam_pp_orig = gradcam_pp(input_tensor=img_tensor.unsqueeze(0))
    gradcam_pp_orig = gradcam_pp_orig[0, :]

    # Grad-CAMPlusPlus for the adversarial example
    gradcam_pp_adv = gradcam_pp(input_tensor=adversarial_example.unsqueeze(0))
    gradcam_pp_adv = gradcam_pp_adv[0, :]

    gradcam_pp_orig = torch.tensor(gradcam_pp_orig).to(device)
    gradcam_pp_adv = torch.tensor(gradcam_pp_adv).to(device)

    return gradcam_pp_orig, gradcam_pp_adv


def create_smoothgrad(model, img_tensor, adversarial_example, device):
    

    return smoothgrad_orig, smoothgrad_adv
