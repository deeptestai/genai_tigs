import os
import numpy as np
from PIL import Image
import cv2

# Directories containing the .npy files for each dataset
vae_input_directory = (
    "/home/maryam/Documents/SEDL/SINVAD/experiments_complete/all_results/mnist_vae_high"
)
gan_input_directory = (
    "/home/maryam/Documents/SEDL/SINVAD/experiments_complete/all_results/mnist_gan_high"
)
sd_input_directory = (
    "/home/maryam/Documents/SEDL/SINVAD/experiments_complete/all_results/mnist_sd_high"
)

# Output directories for each dataset
vae_output_directory = "/home/maryam/Documents/SEDL/SINVAD/experiments_complete/all_results/mnist_vae_high_png"
gan_output_directory = "/home/maryam/Documents/SEDL/SINVAD/experiments_complete/all_results/mnist_gan_high_png"
sd_output_directory = "/home/maryam/Documents/SEDL/SINVAD/experiments_complete/all_results/mnist_sd_high_png"

# Ensure the output directories exist
os.makedirs(vae_output_directory, exist_ok=True)
os.makedirs(gan_output_directory, exist_ok=True)
os.makedirs(sd_output_directory, exist_ok=True)


def convert_and_save(input_directory, output_directory, resize=None):
    for filename in os.listdir(input_directory):
        if filename.endswith(".npy"):
            # Load the .npy file
            file_path = os.path.join(input_directory, filename)
            image_data = np.load(file_path)

            # Check and reshape the data if necessary
            if image_data.shape == (1, 28, 28):
                image_data = image_data.reshape((28, 28))
            elif image_data.shape == (28, 28, 1):
                image_data = image_data.reshape((28, 28))

            # Convert the image data to uint8 (if it's not already)
            if image_data.dtype != np.uint8:
                image_data = (image_data * 255).astype(np.uint8)

            # Resize the image if needed
            if resize:
                image_data = cv2.resize(
                    image_data, resize, interpolation=cv2.INTER_NEAREST
                )

            # Create an Image object from the array
            image = Image.fromarray(image_data, mode="L")  # 'L' mode for grayscale

            # Define the output file path with the same name but .png extension
            output_file_path = os.path.join(
                output_directory, filename.replace(".npy", ".png")
            )

            # Save the image as a PNG file
            image.save(output_file_path)

            print(f"Converted {filename} to {output_file_path}")


# Convert and save for each dataset
convert_and_save(vae_input_directory, vae_output_directory, resize=(224, 224))
convert_and_save(gan_input_directory, gan_output_directory, resize=(224, 224))
convert_and_save(sd_input_directory, sd_output_directory, resize=(224, 224))

print("Conversion complete.")
