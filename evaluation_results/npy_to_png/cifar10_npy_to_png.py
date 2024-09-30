import os
import numpy as np
from PIL import Image
import cv2

# Directories containing the .npy files for each dataset
vae_input_directory = "/add/path/to/input_directory"
gan_input_directory = "/add/path/to/input_directory"
sd_input_directory = "/add/path/to/input_directory"

# Output directories for each dataset
vae_output_directory = "/add/path/to/output_directory"
gan_output_directory = "/add/path/to/output_directory"
sd_output_directory = "/add/path/to/output_directory"

# Ensure the output directories exist
os.makedirs(vae_output_directory, exist_ok=True)
os.makedirs(gan_output_directory, exist_ok=True)
os.makedirs(sd_output_directory, exist_ok=True)


def normalize_image_data(image_data):
    """Normalize the image data to [0, 1] range if it's not already in that range."""
    if image_data.min() < 0 or image_data.max() > 1:
        image_data = (image_data - image_data.min()) / (
            image_data.max() - image_data.min()
        )
    return image_data


def convert_and_save(input_directory, output_directory, resize=None):
    for filename in os.listdir(input_directory):
        if filename.endswith(".npy"):
            # Load the .npy file
            file_path = os.path.join(input_directory, filename)
            image_data = np.load(file_path)
            print(f"Processing {filename}: original shape {image_data.shape}")

            # Check and reshape the data if necessary
            if image_data.shape == (1, 32, 32):
                image_data = image_data.reshape((32, 32))
            elif image_data.shape == (32, 32, 1):
                image_data = image_data.reshape((32, 32))
            elif image_data.shape == (3, 32, 32):
                image_data = np.transpose(
                    image_data, (1, 2, 0)
                )  # Transpose to (32, 32, 3)
            elif image_data.shape == (32, 32, 3):
                pass  # Already in the correct shape

            # Normalize the image data to [0, 1] range
            image_data = normalize_image_data(image_data)

            # Convert the image data to uint8
            image_data = (image_data * 255).astype(np.uint8)

            # Debugging: Print the shape and data type after conversion
            print(
                f"After conversion: shape {image_data.shape}, dtype {image_data.dtype}"
            )

            # Resize the image if needed
            if resize:
                image_data = cv2.resize(
                    image_data, resize, interpolation=cv2.INTER_NEAREST
                )

            # Convert to RGB if it is a color image
            if image_data.ndim == 3 and image_data.shape[2] == 3:
                image = Image.fromarray(image_data, "RGB")
            else:
                image = Image.fromarray(image_data)

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
