'''
MIT License

Copyright (c) 2018 Bjarte Mehus Sunde

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


LINK:  https://github.com/Bjarten/early-stopping-pytorch
====
'''

import numpy as np
import torch


# ----------- Steph-Yves Modified ----------------------------------------------------------- #
# ===========================================================================================
# -------------------------------------------------------------------------------------------

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, increment=0.001,save_best=True,classification=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.classification = classification
        self.patience      = patience
        self.verbose       = verbose
        self.counter       = 0
        self.best_score    = None
        self.early_stop    = False
        self.val_loss_min  = np.Inf

        self.increment     = increment            
        self.flag_value    = f' *** '                 
        self.FLAG          = None        
        self.save_best     = save_best         

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.increment:
            if self.classification == None: self.increase_measure(val_loss, model,score)
            else:                           self.decrease_measure(val_loss, model,score)
        elif score > self.best_score + self.increment:
            if self.classification == None: self.decrease_measure(val_loss, model,score)
            else:                           self.increase_measure(val_loss, model,score)

    def increase_measure(self,val_loss, model,score):
        self.counter += 1
        self.flag_value   = f'> {self.counter} / {self.patience}'
        self.FLAG         = True
        if self.save_best== False:
            self.save_checkpoint(val_loss, model)
        if self.counter >= self.patience:
            self.early_stop = True

    def decrease_measure(self,val_loss, model,score):
        self.best_score = score
        self.save_checkpoint(val_loss, model)
        self.counter = 0
        self.flag_value   = f' *** '

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose: pass
        torch.save(model.state_dict(), 'TRAINED/crystal-checkpoint.pt')
        self.val_loss_min = val_loss
        self.FLAG         = False
