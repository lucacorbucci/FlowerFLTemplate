import torch


def train(net, trainloader, optimizer, device="cpu"):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.to(device)
    net.train()
    losses = 0.0
    correct = 0
    for sample, _, label in trainloader:
        sample, labels = sample.to(device), label.to(device)
        optimizer.zero_grad()
        output = net(sample)
        loss = criterion(output, labels.long())
        loss.backward()
        optimizer.step()
        losses += loss.item()
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == labels).sum().item()

    loss = torch.tensor(losses / len(trainloader), device=device)
    accuracy = correct / len(trainloader.dataset)
    print(f"Train set: Average loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}")
    return loss.item(), accuracy


def test(net, testloader, device):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.to(device)
    net.eval()
    with torch.no_grad():
        for sample, _, label in testloader:
            images, labels = sample.to(device), label.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels.long()).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    print(
        f"Test set: Average loss: {loss / len(testloader):.4f}, Accuracy: {correct}/{len(testloader.dataset)} ({100.0 * accuracy:.2f}%)"
    )
    return float(loss), accuracy
