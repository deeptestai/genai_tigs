import argparse
import os
import subprocess

def main(dataset):
    dataset_paths = {
        'mnist': 'mnist_experiments/vae/train.py',
        'svhn': 'svhn_experiments/vae/train.py',
        'cifar10': 'cifar10_experiments/vae/train.py',
        'imagenet': 'imagenet_experiments/vae/train.py',
    }

    if dataset not in dataset_paths:
        print(f"Dataset {dataset} not recognized. Please choose from 'mnist', 'svhn', 'cifar10', 'imagenet'.")
        return

    script_path = dataset_paths[dataset]

    # Ensure the script path exists
    if not os.path.exists(script_path):
        print(f"Training script for dataset {dataset} not found at {script_path}.")
        return

    # Run the training script
    subprocess.run(['python', script_path])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a VAE model on a specified dataset.')
    parser.add_argument('--dataset', type=str, required=True, help="Dataset to train on. Options: 'mnist', 'svhn', 'cifar10', 'imagenet'")
    args = parser.parse_args()
    
    main(args.dataset)
