from encoder import Encoder

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

import torch
import torch.nn as nn

class LearningRateAdjustment(Callback):
    """
        Callback to adjust the learning rate if validation accuracy drops compared to previous epochs
    """
    def __init__(self, patience=1):
        super().__init__()
        self.patience = patience
        self.recent_accuracy = []

    def on_train_epoch_end(self, trainer, pl_module):

        # get current validation accurancy and learning rate
        val_acc = trainer.callback_metrics['val_acc']
        lr = trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr)

        # adjust learning rate if validation accuracy is lower than recent history
        if len(self.recent_accuracy) > 1:
            if val_acc < min(self.recent_accuracy):
                trainer.optimizers[0].param_groups[0]['lr'] = lr / 5
        
        # add current value and truncate history
        self.recent_accuracy.append(val_acc)
        self.recent_accuracy = self.recent_accuracy[-self.patience:]

        return super().on_train_epoch_end(trainer, pl_module)


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

        optimizer = torch.optim.SGD(self.parameters(), lr=self.opt['lr'], weight_decay=self.opt['weight_decay'])
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda x:0.99),
            'name': 'lr'
        }
        return [optimizer], [lr_scheduler]


    def training_step(self, batch, batch_idx):

        # "batch" is the output of the training data loader.
        (premises, hypotheses), labels = batch
        preds = self.enc(premises, hypotheses)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
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

        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log('test_acc', acc)