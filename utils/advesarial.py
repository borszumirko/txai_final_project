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