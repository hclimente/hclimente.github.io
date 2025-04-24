# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import torch
import torchvision

BATCH_SIZE = 64
DEVICE = "cpu"

# %% [markdown]
# # Load model


# %%
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = torch.nn.Linear(64 * 5 * 5, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleCNN()
model.load_state_dict(torch.load("data/mnist_cnn.pt"))
model.to(DEVICE)
model.eval()

print(model)

# %% [markdown]
# # Compute accuracy on test set

# %%
print("Downloading MNIST test set...")
mnist_te = torchvision.datasets.MNIST(
    "data/mnist",
    train=False,
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)

correct = 0
total = 0

with torch.no_grad():
    for data, target in torch.utils.data.DataLoader(
        mnist_te, batch_size=BATCH_SIZE, shuffle=False
    ):
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f"Accuracy of SimpleCNN on the 10,000 test images: {100 * correct // total} %")
