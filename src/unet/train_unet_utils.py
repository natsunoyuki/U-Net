import torch
import tqdm


def collate_fn(batch):
    """Collate function for torch DataLoader."""
    return tuple(zip(*batch))


def unbatch(batch, device):
    """Unbatches the images and masks in each batch."""
    image, mask = batch
    X = [x.to(device) for x in image]
    y = [y.to(device) for y in mask]
    return X, y


def train_batch(batch, model, optimizer, device):
    """Trains the model on a batch of images and masks."""
    model.train()
    X, y = unbatch(batch, device = device)
    optimizer.zero_grad()
    losses = model(X, y)
    loss = sum(loss for loss in losses.values())
    loss.backward()
    optimizer.step()
    return loss, losses


@torch.no_grad()
def validate_batch(batch, model, optimizer, device):
    """Validates the model on a batch of images and masks."""
    model.train()
    X, y = unbatch(batch, device = device)
    optimizer.zero_grad()
    losses = model(X, y)
    loss = sum(loss for loss in losses.values())
    return loss, losses


def train(model, optimizer, n_epochs, train_loader, test_loader = None, device = torch.device("cpu")):
    model.to(device)
    train_losses = []
    test_losses = []
    for epoch in range(n_epochs):
        N = len(train_loader)
        for i, batch in enumerate(train_loader):
            loss, losses = train_batch(batch, model, optimizer, device)
            train_losses.append([loss, losses])

        if test_loader is not None:
            N = len(test_loader)
            for i, batch in enumerate(test_loader):
                loss, losses = validate_batch(batch, model, optimizer, device)
                test_losses.append([loss, losses])
    
    return model, train_losses, test_losses
