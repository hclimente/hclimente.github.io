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
import matplotlib.pyplot as plt
import sys
import torch
import torchvision

sys.path.append("../../")


BATCH_SIZE = 64
DEVICE = "cpu"
NUM_WORKERS = 0
SEED = 0

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %%
mnist_tr = torchvision.datasets.MNIST(
    "data/mnist",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)

# %%
tr_dataloader = torch.utils.data.DataLoader(
    mnist_tr,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

# plot first 9 examples in a 3x3 grid
it = enumerate(tr_dataloader)
batch_idx, (example_data, example_targets) = next(it)


fig, ax = plt.subplots(3, 3, figsize=(10, 10))
for i in range(9):
    ax[i // 3, i % 3].imshow(example_data[i][0], cmap="gray")
    ax[i // 3, i % 3].set_title(example_targets[i].item())
    ax[i // 3, i % 3].axis("off")

# save_fig(fig, "mnist_examples")


# %% [markdown]
# # Define model


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


model = SimpleCNN().to(DEVICE)
print(model)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

# %% [markdown]
# # Model training

# %%
for batch_idx, (data, target) in enumerate(tr_dataloader):
    data, target = data.to(DEVICE), target.to(DEVICE)
    optimizer.zero_grad()
    output = model(data)  # This should work now
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    if batch_idx % 100 == 0:
        print(f"Train Batch: {batch_idx} Loss: {loss.item():.6f}")  # Simpler print

# %% [markdown]
# # Compute accuracy on test set

# %%
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

torch.save(model.state_dict(), "data/mnist_cnn.pt")
