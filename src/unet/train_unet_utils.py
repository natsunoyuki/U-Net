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


def evaluate_metrics(y_preds, ys, metrics):
    """Evaluates accuracy for each metric in the list metrics."""
    accuracies_per_metric = []
    if metrics is not None:
        for metric in metrics:
            accuracies_per_metric.append(metric(y_preds, ys).cpu().numpy())
    return np.array(accuracies_per_metric) # [acc_1, acc_2, ..., acc_N]


def train_batch(
    batch, 
    model, 
    optimizer, 
    loss_function, 
    metrics=None,
    device=torch.device("cpu"),
):
    """Trains the model on a batch of images and masks."""
    model.train()
    Xs, ys = unbatch(batch, device = device)
    optimizer.zero_grad()
    y_preds = model(Xs)
    loss = loss_function(y_preds, ys)
    loss.backward()
    optimizer.step()

    accuracies = evaluate_metrics(y_preds=y_preds, ys=ys, metrics=metrics)
    return loss.detach().cpu(), accuracies


@torch.no_grad()
def validate_batch(
    batch, 
    model, 
    optimizer, 
    loss_function, 
    metrics=None,
    device=torch.device("cpu"),
):
    """Validates the model on a batch of images and masks."""
    model.train()
    Xs, ys = unbatch(batch, device = device)
    optimizer.zero_grad()
    with torch.no_grad():
        y_preds = model(Xs)
    loss = loss_function(y_preds, ys)

    accuracies = evaluate_metrics(y_preds=y_preds, ys=ys, metrics=metrics)
    return loss.detach().cpu(), accuracies


def dataloader_forward_pass(
    model, 
    optimizer, 
    loss_function, 
    dataloader, 
    forward_batch_function=train_batch, 
    metrics=None,
    device=torch.device("cpu"), 
    verbose=True,
):
    """Runs 1 forward pass epoch."""
    total = len(dataloader)
    losses = []
    epoch_metrics = []

    if verbose is True:
        pbar = tqdm.tqdm(total=total)

    for batch in dataloader:
        batch_loss, batch_accs = forward_batch_function(
            batch=batch, 
            model=model, 
            optimizer=optimizer, 
            loss_function=loss_function, 
            device=device,
            metrics=metrics,
        )

        losses.append(batch_loss)
        epoch_metrics.append(batch_accs)

        if verbose is True:
            # Print current batch loss and first accuracy.
            if len(batch_accs) == 0:
                t = "Loss={:.2f}".format(batch_loss)
            else:
                t = "Loss={:.2f}, Acc={:.2f}".format(batch_loss, batch_accs[0])
            pbar.set_description(t)
            pbar.update()

    # Losses and epoch_metrics will have length equal to the number of batches.
    return losses, epoch_metrics


def train(
    model, 
    optimizer, 
    loss_function, 
    n_epochs, 
    train_loader, 
    test_loader=None, 
    metrics=None,
    device=torch.device("cpu"), 
    verbose=False,
):
    """Trains a model over n_epochs with an optimizer, loss function and data 
    loaders."""
    model.to(device)

    if metrics is not None:
        for metric in metrics:
            metric.to(device)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(n_epochs):
        if verbose is True:
            print("================ Epoch {:04d} ================".format(
                    epoch + 1
                )
            )

        l, a = dataloader_forward_pass(
            model=model, 
            optimizer=optimizer, 
            loss_function=loss_function, 
            dataloader=train_loader, 
            forward_batch_function=train_batch, 
            metrics=metrics,
            device=device, 
            verbose=verbose,
        )
        # l.shape = (n_batches,), a.shape = (n_batches, n_metrics).
        train_losses.append(np.mean(l))
        if metrics is not None:
            train_accuracies.append(np.mean(a, axis = 0))

        if test_loader is not None:
            l, a = dataloader_forward_pass(
                model=model, 
                optimizer=optimizer, 
                loss_function=loss_function, 
                dataloader=test_loader, 
                forward_batch_function=validate_batch, 
                metrics=metrics,
                device=device, 
                verbose=verbose,
            )
            test_losses.append(np.mean(l))
            if metrics is not None:
                test_accuracies.append(np.mean(a, axis = 0))

        if verbose is True:
            print("Train loss: {:.3f}. Test loss: {:.3f}.".format(
                    train_losses[-1], test_losses[-1],
                )
            )
    
    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)
    train_accuracies = np.array(train_accuracies)
    test_accuracies = np.array(test_accuracies)

    return model, train_losses, test_losses, train_accuracies, test_accuracies
