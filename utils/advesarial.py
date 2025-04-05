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

def create_PGD_example(model, img_tensor, classindex, delta, device, iterations, epsilon):
    label = torch.Tensor([classindex]).long()
    label = label.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    img_tensor_copy = img_tensor.clone().to(device)
    img_tensor.requires_grad = True
    
    for i in range(iterations):
        outputs = model(img_tensor.unsqueeze(0))
        loss = loss_fn(outputs, label)

        model.zero_grad()
        loss.backward()

        adversarial_example = img_tensor_copy + delta*torch.sign(img_tensor.grad)
        perturbation = torch.clamp(adversarial_example - img_tensor_copy, min=-epsilon, max=epsilon)
        adversarial_example = torch.clamp(img_tensor_copy + perturbation, min=0, max=1)

    return adversarial_example, img_tensor_copy


