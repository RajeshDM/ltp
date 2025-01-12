from pytorch_lightning.callbacks import Callback

import  pytorch_lightning as pl

class ValidationLossLogging(Callback):
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        validation_loss = float(trainer.callback_metrics['validation_loss'])
        print('[{}] Validation loss: {}'.format(trainer.current_epoch, validation_loss))