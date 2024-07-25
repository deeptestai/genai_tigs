import argparse
import os
import subprocess

def main(dataset):
    dataset_paths = {
        'mnist': 'mnist_experiments/mnist_gan/cdcgan/sinvad_cdcgan_mnist.py',
        'svhn': 'svhn_experiments/svhn_gan/cdcgan/sinvad_cdcgan_svhn.py',
        'cifar10': 'cifar10_experiments/cifar10_gan/cdcgan/sinvad_cdcgan_cifar10.py',
    }

    if dataset not in dataset_paths:
        print(f"Dataset {dataset} not recognized. Please choose from 'mnist', 'svhn', 'cifar10'.")
        return

    script_path = dataset_paths[dataset]

    # Ensure the script path exists
    if not os.path.exists(script_path):
        print(f"GAN script for dataset {dataset} not found at {script_path}.")
        return

    # Run the GAN script
    subprocess.run(['python', script_path])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a SINVAD CDCGAN model on a specified dataset.')
    parser.add_argument('--dataset', type=str, required=True, help="Dataset to use. Options: 'mnist', 'svhn', 'cifar10'")
    args = parser.parse_args()
    
    main(args.dataset)
