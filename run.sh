#!/bin/bash

echo "🔧 Building Docker image..."
docker build -t giftbench-app .

docker run -d --gpus all -p 7860:7860 -v "$PWD":/app -w /app --name giftbench-running giftbench-app && docker logs -f giftbench-running


