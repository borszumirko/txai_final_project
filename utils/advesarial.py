import torch

def create_FGSM_example(model, img_tensor, classindex, delta, device):
    label = torch.Tensor([classindex]).long()
    label = label.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    img_tensor_copy = img_tensor.clone().to(device)
    img_tensor.requires_grad = True
    outputs = model(img_tensor.unsqueeze(0))

    loss = loss_fn(outputs, label)
    loss.backward()

    delta = .03
    adversarial_example = img_tensor_copy + delta * torch.sign(img_tensor.grad)
    return adversarial_example, img_tensor_copy

# From tutorial code
def create_PGD_example(model, img_tensor, classindex, device, alpha=0.005, delta=0.03, num_iterations=10):
    label = torch.Tensor([classindex]).long()
    label = label.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    img_tensor_copy = img_tensor.clone().to(device)
    img_tensor.requires_grad = True
    
    pgd_adverserial_images = []

    # Perform the PGD attack for the specified number of iterations
    for i in range(num_iterations):
        outputs = model(img_tensor.unsqueeze(0))
        loss = loss_fn(outputs, label)
        loss.backward()
        
        # Update the adversarial image
        img_tensor = img_tensor + alpha * torch.sign(img_tensor.grad)
        
        # Clamp the values to ensure the image is within the valid range
        img_tensor = torch.clamp(img_tensor, img_tensor_copy - delta, img_tensor_copy + delta)
        
        # Append the current adversarial image to the list
        pgd_adverserial_images.append(img_tensor)
        
        # Detach the gradient to prevent accumulation in the next iteration
        img_tensor = img_tensor.detach()
        img_tensor.requires_grad = True

    # Return the last element of the adversarial images list
    return pgd_adverserial_images[-1], img_tensor_copy
