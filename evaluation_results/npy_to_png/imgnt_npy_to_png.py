import os
import numpy as np
import cv2

# Directories containing the .npy files for each dataset
vae1_input_directory = "/add/path/to/input_directory/Imagenetteddy_vae_low"
vae2_input_directory = "/add/path/to/input_directory/Imagenetteddy_vae_high"
gan1_input_directory = "/add/path/to/input_directory/Imagenetteddy_gan_low"
gan2_input_directory = "/add/path/to/input_directory/Imagenetteddy_gan_high"
sd1_input_directory = "/add/path/to/input_directory/Imagenetteddy_sd_low"
sd2_input_directory = "/add/path/to/input_directory/Imagenetteddy_sd_high"

# Output directories for each dataset
vae1_output_directory = "/add/path/to/onput_directory/Imagenetteddy_vae_low_png"
vae2_output_directory = "/add/path/to/onput_directory/Imagenetteddy_vae_high_png"
gan1_output_directory = "/add/path/to/onput_directory/Imagenetteddy_gan_low_png"
gan2_output_directory = "/add/path/to/onput_directory/Imagenetteddy_gan_high_png"
sd1_output_directory = "/add/path/to/onput_directory/Imagenetteddy_sd_low_png"
sd2_output_directory = "/add/path/to/onput_directory/Imagenetteddy_sd_high_png"

# Ensure the output directories exist
os.makedirs(vae1_output_directory, exist_ok=True)
os.makedirs(vae2_output_directory, exist_ok=True)
os.makedirs(gan1_output_directory, exist_ok=True)
os.makedirs(gan2_output_directory, exist_ok=True)
os.makedirs(sd1_output_directory, exist_ok=True)
os.makedirs(sd2_output_directory, exist_ok=True)


def normalize_image_data(image_data):
    """Normalize the image data to [0, 1] range if it's not already in that range."""
    if image_data.min() < 0 or image_data.max() > 1:
        image_data = (image_data - image_data.min()) / (
            image_data.max() - image_data.min()
        )
    return image_data


def convert_and_save(input_directory, output_directory, resize=(224, 224)):
    for filename in os.listdir(input_directory):
        if filename.endswith(".npy"):
            try:
                # Load the .npy file
                file_path = os.path.join(input_directory, filename)
                image_data = np.load(file_path)
                print(f"Processing {filename}: original shape {image_data.shape}")

                # Check and reshape the data if necessary
                if image_data.shape == (1, 224, 224):
                    image_data = image_data.reshape((224, 224))
                elif image_data.shape == (224, 224, 1):
                    image_data = image_data.reshape((224, 224))
                elif image_data.shape == (3, 224, 224):
                    image_data = np.transpose(
                        image_data, (1, 2, 0)
                    )  # Transpose to (224, 224, 3)
                elif image_data.shape == (224, 224, 3):
                    pass  # Already in the correct shape
                elif image_data.shape[0] == 3:  # Handle images of shape (3, H, W)
                    image_data = np.transpose(
                        image_data, (1, 2, 0)
                    )  # Transpose to (H, W, 3)
                else:
                    print(f"Unexpected image shape {image_data.shape} for {filename}")
                    continue

                # Normalize the image data to [0, 1] range
                image_data = normalize_image_data(image_data)

                # Convert the image data to uint8
                image_data = (image_data * 255).astype(np.uint8)

                # Debugging: Print the shape and data type after conversion
                print(
                    f"After conversion: shape {image_data.shape}, dtype {image_data.dtype}"
                )

                # Resize the image if needed
                image_data = cv2.resize(
                    image_data, resize, interpolation=cv2.INTER_NEAREST
                )

                # Define the output file path with the same name but .png extension
                output_file_path = os.path.join(
                    output_directory, filename.replace(".npy", ".png")
                )

                # Save the image as a PNG file using OpenCV
                if len(image_data.shape) == 2:  # Grayscale image
                    cv2.imwrite(output_file_path, image_data)
                elif (
                    len(image_data.shape) == 3 and image_data.shape[2] == 3
                ):  # RGB image
                    cv2.imwrite(
                        output_file_path, cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
                    )
                else:
                    print(f"Unexpected image shape {image_data.shape} for {filename}")

                print(f"Converted {filename} to {output_file_path}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")


# Convert and save for each dataset
convert_and_save(vae1_input_directory, vae1_output_directory)
convert_and_save(vae2_input_directory, vae2_output_directory)
convert_and_save(gan1_input_directory, gan1_output_directory)
convert_and_save(gan2_input_directory, gan2_output_directory)
convert_and_save(sd1_input_directory, sd1_output_directory)
convert_and_save(sd2_input_directory, sd2_output_directory)

print("Conversion complete.")
