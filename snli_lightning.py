from encoder import Encoder

import pytorch_lightning as pl
from timeit import default_timer as timer

import torch
import torch.nn as nn

class SNLIModule(pl.LightningModule):

    def __init__(self, embedding, opt):

        super().__init__()
        self.opt = opt
        self.enc = Encoder(embedding, opt)
        self.loss_module = nn.CrossEntropyLoss()


    def forward(self, x):

        premises, hypotheses = x
        return self.enc(premises, hypotheses)


    def configure_optimizers(self):

        optimizer = torch.optim.SGD(self.parameters(), lr=self.opt['lr'])
        scheduler1 = {
            "scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.opt["weight_decay"]),
            "interval": "epoch"
        }
        scheduler2 = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode="min", 
                factor=0.2, 
                patience=self.opt["patience"]
            ), 
            "interval": "epoch", 
            "monitor": "val_acc"
        }

        return [optimizer], [scheduler1, scheduler2]

            
    def training_step(self, batch, batch_idx):

        # "batch" is the output of the training data loader.
        start = timer()
        (premises, hypotheses), labels = batch
        preds = self.enc(premises, hypotheses)
        pred_time = timer()
        loss = self.loss_module(preds, labels)
        loss_time = timer()
        acc = (preds.argmax(dim=-1) == labels).float().mean()


        # Logs the accuracy to tensorboard
        self.log('train_acc', acc, on_step=True)
        self.log('train_loss', loss, on_step=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)
        log_time = timer()
        # print("Training step takes: {:7.2} milliseconds".format(1000*(log_time-start)))
        return loss


    def validation_step(self, batch, batch_idx):

        (premises, hypotheses), labels = batch
        preds = self.enc(premises, hypotheses).argmax(dim=-1)
        acc = (labels == preds).float().mean()

        # By default logs it per epoch (weighted average over batches)
        self.log('val_acc', acc)


    def test_step(self, batch, batch_idx):

        (premises, hypotheses), labels = batch
        preds = self.enc(premises, hypotheses).argmax(dim=-1)
        acc = (labels == preds).float().mean()

        # By default logs it per epoch (weighted average over batches)
        self.log('test_acc', acc)
