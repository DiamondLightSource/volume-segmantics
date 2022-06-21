import logging

import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', model_dict={}, best_score=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.val_loss_min = np.inf if best_score is None else best_score * -1
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.model_struc_dict = model_dict # Dictionary with parameters controlling architecture

    def __call__(self, val_loss, model, optimizer, label_codes):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, label_codes)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logging.info(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, label_codes)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, label_codes):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logging.info(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        model_dict = {
            "model_state_dict": model.state_dict(),
            "model_struc_dict": self.model_struc_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_val": val_loss,
            "label_codes": label_codes,
        }        
        torch.save(model_dict, self.path)
        self.val_loss_min = val_loss
