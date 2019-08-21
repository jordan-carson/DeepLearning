from sklearn.datasets import load_wine
from torchvision.datasets import FashionMNIST

data = load_wine()

print((data.target))