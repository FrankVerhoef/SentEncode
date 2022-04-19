import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import datetime


from torch.utils.tensorboard import SummaryWriter


def train(model, train_loader, valid_loader, opt):

    # initialize logger
    now = datetime.datetime.now()
    datestr = now.strftime("%Y%m%d_%H%M")
    logger = SummaryWriter("logs/log_" + datestr)

    # select device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # define optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=opt['lr'], weight_decay=opt['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=1, verbose=True)

    # Training loop with validation after each epoch, saving the best model
    best_accuracy = -1.0

    for epoch in range(opt["max_epochs"]):

        model.train()

        batch_idx = 0
        for batch in tqdm(iter(train_loader), desc="Train batch"):

            # get batch
            ((premises, p_len), (hypotheses, h_len)), labels = batch
            premises = premises.to(device)
            hypotheses = hypotheses.to(device)
            labels = labels.to(device)

            # calculate predictions and calculate metrics
            preds = model((premises, p_len), (hypotheses, h_len))
            loss = criterion(preds, labels)
            acc = (preds.argmax(dim=-1) == labels).float().mean()

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log loss, accuracy and learning rate per step
            step = epoch * len(batch) + batch_idx
            logger.add_scalar('train_acc', acc, global_step=step)
            logger.add_scalar('train_loss', loss, global_step=step)
            logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=step)
            batch_idx += 1

        # Check accuracy validation set, save best model
        model.eval()    
        val_accuracy = evaluate_model(model, valid_loader, device)
        logger.add_scalar("valid_acc", val_accuracy, global_step=step)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            state_dict = model.state_dict()
            torch.save(state_dict, opt["checkpoint_name"] + "_" + opt["encoder_type"] + "_" + str(epoch))

        lr_scheduler.step(val_accuracy)
        if optimizer.param_groups[0]['lr'] < opt['lr_limit']:
            break

    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.
    """

    model.to(device)

    # Initialise lists to store accuracy and size of each batch
    batch_accuracy = []
    batch_size = []

    # Loop through the whole dataset, per batch
    for batch in tqdm(iter(data_loader), desc="Valid batch"):

        ((premises, p_len), (hypotheses, h_len)), labels = batch
        premises = premises.to(device)
        hypotheses = hypotheses.to(device)
        labels = labels.to(device)

        # Get the predictions
        predictions = model((premises, p_len), (hypotheses, h_len)).argmax(dim=1)
        acc = (labels == predictions).float().mean()

        # Append the results
        batch_accuracy.append(acc.cpu())
        batch_size.append(len(labels))

    avg_accuracy = np.inner(batch_accuracy, batch_size) / sum(batch_size)

    return avg_accuracy


