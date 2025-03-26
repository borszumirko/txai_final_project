import torch
from pytorch_grad_cam import GradCAM

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