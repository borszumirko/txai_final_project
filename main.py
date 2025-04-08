import torch
from experiment import run_experiment


def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    config_filename = "config.json"
    
    run_experiment(config_filename, device)

if __name__ == "__main__":
    main()