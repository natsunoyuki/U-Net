import torch
import tqdm
import numpy as np


def collate_fn(batch):
    """Collate function for torch DataLoader."""
    Xs, ys = zip(*batch)
    return torch.stack(Xs), torch.stack(ys)


def unbatch(batch, device=torch.device("cpu")):
    """Unbatches the images and masks in each batch."""
    images, masks = batch
    return images.to(device), masks.to(device)


def train_batch(
    batch, model, optimizer, loss_function, device=torch.device("cpu")
):
    """Trains the model on a batch of images and masks."""
    model.train()
    Xs, ys = unbatch(batch, device = device)
    optimizer.zero_grad()
    y_preds = model(Xs)
    loss = loss_function(y_preds, ys)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def validate_batch(
    batch, model, optimizer, loss_function, device=torch.device("cpu")
):
    """Validates the model on a batch of images and masks."""
    model.train()
    Xs, ys = unbatch(batch, device = device)
    optimizer.zero_grad()
    with torch.no_grad():
        y_preds = model(Xs)
    loss = loss_function(y_preds, ys)
    return loss


def train(
    model, 
    optimizer, 
    loss_function, 
    n_epochs, 
    train_loader, 
    test_loader=None, 
    device=torch.device("cpu"), 
    verbose=False,
):
    """Trains a model over n_epochs with an optimizer, loss function and data 
    loaders."""
    model.to(device)
    train_losses = []
    test_losses = []
    
    for epoch in range(n_epochs):
        if verbose is True:
            print("================ Epoch {:04d} ================".format(
                epoch + 1)
            )

        losses = dataloader_forward_pass(
            model, 
            optimizer, 
            loss_function, 
            train_loader, 
            train_batch, 
            device, 
            verbose,
        )
        train_losses.append(np.mean(losses))

        if test_loader is not None:
            losses = dataloader_forward_pass(
                model, 
                optimizer, 
                loss_function, 
                test_loader, 
                validate_batch, 
                device, 
                verbose,
            )
            test_losses.append(np.mean(losses))

        if verbose is True:
            print("Train loss: {:.3f}. Test loss: {:.3f}.".format(
                train_losses[-1], test_losses[-1])
            )
    
    return model, train_losses, test_losses


def dataloader_forward_pass(
    model, 
    optimizer, 
    loss_function, 
    dataloader, 
    forward_batch_function=train_batch, 
    device=torch.device("cpu"), 
    verbose=True,
):
    total = len(dataloader)
    losses = []

    if verbose is True:
        pbar = tqdm.tqdm(total=total)
    
    for i, batch in enumerate(dataloader):
        loss = forward_batch_function(
            batch, model, optimizer, loss_function, device,
        )
        losses.append(loss.detach().cpu())

        if verbose is True:
            pbar.set_description("Loss={:.2f}".format(loss.detach().cpu()))
            pbar.update()

    return losses
