import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, ScoreCAM, LayerCAM
from captum.attr import LRP

# General function for calculating a GradCam variant
def create_cam(CamClass, model, img_tensor, adversarial_example, device):
    cam = CamClass(model=model, target_layers=[model.layer4[2]])
    cam_orig = cam(input_tensor=img_tensor.unsqueeze(0))
    cam_orig = cam_orig[0, :]

    cam_adv = cam(input_tensor=adversarial_example.unsqueeze(0))
    cam_adv = cam_adv[0, :]

    cam_orig = torch.tensor(cam_orig).to(device)
    cam_adv = torch.tensor(cam_adv).to(device)

    return cam_orig, cam_adv



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


def create_eigencam(model, img_tensor, adversarial_example, device):
    
    eigencam = EigenCAM(model=model, target_layers=[model.layer4[0]])
    eigencam_orig = eigencam(input_tensor=img_tensor.unsqueeze(0))
    eigencam_orig = eigencam_orig[0, :]

    eigencam_adv = eigencam(input_tensor=adversarial_example.unsqueeze(0))
    eigencam_adv = eigencam_adv[0, :]

    eigencam_orig = torch.tensor(eigencam_orig).to(device)
    eigencam_adv = torch.tensor(eigencam_adv).to(device)

    return eigencam_orig, eigencam_adv


def create_lrp(model, img_tensor, adversarial_example, device, pred_orig, pred_adv):
    
    lrp = LRP(model)
    
    attributions_orig = lrp.attribute(img_tensor.unsqueeze(0), target=pred_orig).to(device)
    attributions_adv = lrp.attribute(adversarial_example.unsqueeze(0), pred_adv).to(device)
    
    attributions_orig = attributions_orig[0, :]
    attributions_adv = attributions_adv[0, :]

    return attributions_orig, attributions_adv

def create_layercam(model, img_tensor, adversarial_example, device):
    
    layercam = LayerCAM(model=model, target_layers=[model.layer4[0]])
    layercam_orig = layercam(input_tensor=img_tensor.unsqueeze(0))
    layercam_orig = layercam_orig[0, :]

    layercam_adv = layercam(input_tensor=adversarial_example.unsqueeze(0))
    layercam_adv = layercam_adv[0, :]

    layercam_orig = torch.tensor(layercam_orig).to(device)
    layercam_adv = torch.tensor(layercam_adv).to(device)

    return layercam_orig, layercam_adv


def calculate_saliency(model, img_tensor, adversarial_example, device, pred_orig, pred_adv, heatmap_type='gradcam'):
    saliency_functions = {
        "gradcam": lambda: create_gradcam(model, img_tensor, adversarial_example, device),
        "gradcam++": lambda: create_gradcam_pp(model, img_tensor, adversarial_example, device),
        "eigen": lambda: create_eigencam(model, img_tensor, adversarial_example, device),
        "layer": lambda: create_layercam(model, img_tensor, adversarial_example, device),
        "lrp": lambda: create_lrp(model, img_tensor, adversarial_example, device, pred_orig, pred_adv),
    }

    if heatmap_type not in saliency_functions:
        raise ValueError(f"Unknown heatmap: {heatmap_type}")

    return saliency_functions[heatmap_type]()


'''
def create_smoothgrad(model, img_tensor, adversarial_example, classindex, device):
    
    smooth_orig = compute_smooth_gradient(model, img_tensor, classindex, delta=0.2, samples=10, device=device)
    smooth_adv = compute_smooth_gradient(model, adversarial_example, classindex, delta=0.2, samples=10, device=device)
    
    return smoothgrad_orig, smoothgrad_adv

#computes the smoothgradients
def compute_smoothgrad(model, img_tensor, classindex, delta, samples, device):
    label = torch.Tensor([classindex]).long()
    label = label.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    img_tensor_copy = img_tensor.clone().to(device)
    img_tensor.requires_grad = True
    total_gradients = torch.zeros_like(img_tensor_copy)

    #gradient is averaged from samples, samples are made by adding noise to the original
    for i in range(samples):
        noisy_img = image + delta * torch.randn_like(img_tensor)
        noisy_img.requires_grad = True

        outputs = model(noisy_img.unsqueeze(0))
        loss = loss_fn(outputs, label)

        model.zero_grad()
        loss.backward()

        total_gradients += noisy_img.grad.data

    smooth_grad = total_gradients/samples
    smooth_grad = smooth_grad.abs().squeeze()

    if smooth_grad.ndim == 3:
        smooth_grad = smooth_grad.max(dim=0)[0]

    return smoothgrad

'''