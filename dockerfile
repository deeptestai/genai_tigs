# Use a CUDA-compatible PyTorch image
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set working directory inside container
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    unzip \
    && rm -rf /var/lib/apt/lists/*


# Copy app code
COPY . .

# Install Python dependencies including gdown
RUN pip install --upgrade pip && pip install -r requirements.txt gdown

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Expose port for Gradio
EXPOSE 7860

# Run entrypoint (downloads models + runs app)
CMD ["./entrypoint.sh"]
