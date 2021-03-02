import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.serialization import normalize_storage_type
import torchvision
from torchvision import transforms
import time

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(1, 32, 5, padding=2)
        self.conv2=nn.Conv2d(32, 64, 5, padding=2)
        self.fc1=nn.Linear(64 * 7 * 7, 64)
        self.fc2=nn.Linear(64, 32)
        self.fc3=nn.Linear(32, 10)
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x=x.view(-1,64*7*7)
        x=F.relu(self.fc1(x))
        x=F.dropout(x, 0.5, training=True)
        x=F.relu(self.fc2(x))
        x=F.softmax(self.fc3(x))
        return x

def main():
    transform=transforms.Compose([transforms.ToTensor()])
    train_set=torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader=torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=0
    )
    test_set=torchvision.datasets.MNIST(
        root="./data", train=False,download=True, transform=transform
    )
    test_loader=torch.utils.data.DataLoader(
        test_set, batch_size=4, shuffle=False, num_workers=0
    ) 
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net=Net()

    optimizer = optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    net.to(device)
    net.train()
    
    torch.cuda.synchronize()
    ts = time.time()
    for epoch in range(10):
        for batch_idx,(data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output=net(data)
            loss=criterion(output, target)
            loss.backward()
            optimizer.step()

            if(batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch: {epoch} | Batch: {batch_idx+1} | Loss: {loss.item():.6f}"
                )

    torch.cuda.synchronize()
    print(f"Train time {time.time() - ts:.2f}")

    net.eval()
    correct_count=0
    torch.cuda.synchronize()
    ts=time.time()
    with torch.no_grad():
        for data,target in test_loader:
            data, target = data.to(device), target.to(device)
            output=net(data)
            pred=output.argmax(dim=1, keepdim=True)
            correct_count+=pred.eq(target.view_as(pred)).sum().item()
    torch.cuda.synchronize()
    print(f"\nTest time{time.time() - ts:.2f}")
    print(
        f"Accuracy: {100.0 * correct_count / len(test_loader.dataset)}%({correct_count}/{len(test_loader.dataset)})"    
    )
    
    torch.save(net.state_dict(), "./exr.pt")

    net_ = Net()
    net_.load_state_dict(torch.load("./exr.pt"))
    net_tsp = torch.jit.script(net_)
    net_tsp.save("./exr_tsp.pt")


if __name__=="__main__":
    main()
