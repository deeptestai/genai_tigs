import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import sys

from model import VGGNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
result_dir = "runtime_results"
os.makedirs(result_dir, exist_ok=True)

# Hyper-parameters
img_size = 32 * 32 * 1
h_dim = 800
num_epochs = 100
batch_size = 128
learning_rate = 2e-4

# SVHN dataset
train_dataset = torchvision.datasets.SVHN(
    root="../data", split="train", transform=transforms.ToTensor(), download=True
)
test_dataset = torchvision.datasets.SVHN(
    root="../data", split="test", transform=transforms.ToTensor(), download=True
)

# Data loader
train_data_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)
test_data_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)

sys.argv[1] == "vgg"
model = VGGNet()
# elif sys.argv[1] == "mobile":
#  model = MobileNet()
# elif sys.argv[1] == "custom":
#  model = CifarClassifier()
# else:
# raise ValueError(f"Unknown network type {sys.argv[0]}")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for i, (x, x_class) in enumerate(train_data_loader):
        # Forward pass
        x = x.to(device)  # .view(-1, img_size)
        class_logits = model(x)

        # Backprop and optimize
        loss = loss_fn(class_logits, x_class.to(device))

        # darc1 regularizer (optional)
        darc1_loss = 0  # 1e-3*torch.max(torch.sum(torch.abs(class_logits), dim=0))
        loss = darc1_loss + loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                "Epoch[{}/{}], Step [{}/{}], CE Loss: {:.4f}".format(
                    epoch + 1, num_epochs, i + 1, len(train_data_loader), loss.item()
                )
            )

    with torch.no_grad():
        total = 0.0
        correct = 0.0
        for tx, tx_class in test_data_loader:
            tx = tx.to(device)  # .view(-1, img_size)
            tclass_logits = model(tx)
            _, mostprob_result = torch.max(tclass_logits, dim=1)
            total += tx.size(0)
            correct += torch.sum(mostprob_result == tx_class.to(device))
        print(
            "%d/%d correct (%.2f %%)" % (correct, total, 100 * float(correct) / total)
        )
model_save_path = os.path.join(result_dir, f"SVHN_{sys.argv[1]}net.pth")
torch.save(model.state_dict(), model_save_path)
