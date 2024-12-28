import torch

from architecture import AddModelBase
from torch.nn.functional import l1_loss


class AddModel(AddModelBase):
    def __init__(self, predicates: list, hidden_size: int, iterations: int, learning_rate: float, l1_factor: float, weight_decay: float):
        super().__init__(predicates, hidden_size, iterations)
        self.save_hyperparameters('learning_rate', 'l1_factor', 'weight_decay')
        self.learning_rate = learning_rate
        self.l1_factor = l1_factor
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, train_batch, batch_index):
        states, target = train_batch
        output = self(states)
        loss = torch.mean(torch.abs(torch.sub(target, output)))
        self.log('train_loss', loss)
        if self.l1_factor > 0.0:
            l1_loss = 0.0
            for parameter in self.parameters():
                l1_loss += torch.sum(self.l1_factor * torch.abs(parameter))
            self.log('l1_loss', l1_loss)
            loss += l1_loss
        self.log('total_loss', loss)
        return loss

    def validation_step(self, validation_batch, batch_index):
        states, target = validation_batch
        output = self(states)
        loss = l1_loss(output, target)
        self.log('validation_loss', loss)