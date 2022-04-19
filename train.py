from encoder import Encoder

import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F

class SNLIModule(pl.LightningModule):

    def __init__(self, embedding, opt):

        super().__init__()
        self.opt = opt
        self.enc = Encoder(embedding, opt)
        # self.loss_module = nn.CrossEntropyLoss()


    def forward(self, x):

        premises, hypotheses = x
        return self.enc(premises, hypotheses)


    def configure_optimizers(self):

        optimizer = torch.optim.SGD(self.parameters(), lr=self.opt['lr'], weight_decay=self.opt['weight_decay'])
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, verbose=True), 
            "monitor": "val_acc",
            "name": 'lr'
        }
        return [optimizer], [lr_scheduler]

            
    def training_step(self, batch, batch_idx):

        # "batch" is the output of the training data loader.
        (premises, hypotheses), labels = batch
        preds = self.enc(premises, hypotheses)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc', acc, on_step=True)
        self.log('train_loss', loss, on_step=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)
        return loss


    def validation_step(self, batch, batch_idx):

        (premises, hypotheses), labels = batch
        preds = self.enc(premises, hypotheses).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.lr_schedulers().step(acc)

        # By default logs it per epoch (weighted average over batches)
        self.log('val_acc', acc, on_step=True)


    def test_step(self, batch, batch_idx):

        (premises, hypotheses), labels = batch
        preds = self.enc(premises, hypotheses).argmax(dim=-1)
        acc = (labels == preds).float().mean()

        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log('test_acc', acc)
