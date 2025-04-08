import torch

def calculate_iou(saliency_map_1, saliency_map_2, threshold=0.75):

    # Normalize the saliency maps to the range [0, 1]
    saliency_map_1 = normalize_tensor(saliency_map_1)
    saliency_map_2 = normalize_tensor(saliency_map_2)

    binary_map_1 = (saliency_map_1 > threshold).int()
    binary_map_2 = (saliency_map_2 > threshold).int()
    
    intersection = torch.sum(binary_map_1 & binary_map_2)
    
    union = torch.sum(binary_map_1 | binary_map_2)
    
    iou = intersection.float() / union.float() if union != 0 else torch.tensor(0.0).to(saliency_map_1.device)
    return iou

def squared_difference(saliency_map_1, saliency_map_2):
    return torch.sum((saliency_map_1 - saliency_map_2) ** 2)

def absolute_difference(saliency_map_1, saliency_map_2):
    return torch.sum(torch.abs(saliency_map_1 - saliency_map_2))

def cosine_similarity(saliency_map_1, saliency_map_2):
    return torch.nn.functional.cosine_similarity(saliency_map_1.flatten(), saliency_map_2.flatten(), dim=0)

def calculate_distance(saliency_map_1, saliency_map_2, distance_metric='cosine'):
    distance_functions = {
        "cosine": lambda: cosine_similarity(saliency_map_1, saliency_map_2),
        "squared": lambda: squared_difference(saliency_map_1, saliency_map_2),
        "absolute": lambda: absolute_difference(saliency_map_1, saliency_map_2),
        "iou_50": lambda: calculate_iou(saliency_map_1, saliency_map_2, threshold=0.50),
        "iou_65": lambda: calculate_iou(saliency_map_1, saliency_map_2, threshold=0.65),
        "iou_70": lambda: calculate_iou(saliency_map_1, saliency_map_2, threshold=0.70),
        "iou_75": lambda: calculate_iou(saliency_map_1, saliency_map_2, threshold=0.75),
        "iou_80": lambda: calculate_iou(saliency_map_1, saliency_map_2, threshold=0.80),
        "iou_90": lambda: calculate_iou(saliency_map_1, saliency_map_2, threshold=0.90),
        "iou_95": lambda: calculate_iou(saliency_map_1, saliency_map_2, threshold=0.95),
    }

    if distance_metric not in distance_functions:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    return distance_functions[distance_metric]()  # Call the corresponding function


def normalize_tensor(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized = (tensor - min_val) / (max_val - min_val + 1e-10) # Avoid divide by 0  
    return normalized
