import torch 

import numpy as np
from tqdm import tqdm  
from datetime import datetime, timedelta
import time
import os 

from loggers import TensorboardLogger

class Trainer():
    def __init__(self, model, 
                 device,
                 dataloader,
                 loss, 
                 optimizer,
                 scheduler,
                 config):

        self.config = config
        self.device = device
        
        self.model = model
        self.train_loader, self.val_loader = dataloader
        self.loss = loss
        #self.metric = metric
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_grads = self.config['dataset']['train']['clip_grads']
        
        # Train ID 
        self.train_id = self.config['id']
        self.train_id += ('-' + 'ResNet50' + '-' + self.config['dataset']['name'] + '-' +  
                         'lr:' + str(self.config['trainer']['lr']) + '-' + 
                         datetime.now().strftime('%Y_%m_%d-%H_%M_%S') +
                         '-timm') 

        self.save_dir = os.path.join('checkpoints', self.train_id)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # Logger
        self.tsboard = TensorboardLogger(path=self.save_dir)

        # Get arguments
        self.nepochs = self.config['trainer']['nepochs']
        self.log_step = self.config['trainer']['log_step']
        self.val_step = self.config['trainer']['val_step']
        
        self.best_loss = np.inf
        self.best_metric = 0.0
        self.val_loss = list()
        

    def train_epoch(self, epoch, iterator):
        print('Training........')

        # 0: Record loss during training process
        running_loss = []
        total_loss = []

        # Switch model to training mode
        self.model.train()

        # Setup progress bar
        progress_bar = tqdm(iterator)
        for i, (img, lbl) in enumerate(progress_bar):
            # 1: Load sources, targets
            img = img.to(self.device)
            lbl = lbl.to(self.device)

            # 4: Clear gradients from previous iteration
            self.optimizer.zero_grad()

            # 2: Predict
            out = self.model(img)
            #print(torch.isfinite(out))
            # 3: Calculate the loss
            loss = self.loss(out, lbl)           
        
            # 5: Calculate gradients
            loss.backward()
            
            # 6: Performing backpropagation
            self.optimizer.step()
            
            with torch.no_grad():
                # 7: Update loss
                running_loss.append(loss.item())
                total_loss.append(loss.item())
                
                # Update loss every log_step or at the end
                if i % self.log_step == 0 or (i + 1) == len(iterator):
                    self.tsboard.update_loss(
                            'train', 
                            sum(running_loss) / len(running_loss), 
                            epoch * len(iterator) + i)
                    running_loss.clear()
            
        print('++++++++++++++ Training result ++++++++++++++')
        avg_loss = sum(total_loss) / len(iterator)
        print('Loss: ', avg_loss)
        return avg_loss  
    
    @torch.no_grad()
    def val_epoch(self, epoch, iterator):
        print('Evaluating........')

        # 0: Record loss during training process
        total_loss = []
        total_acc = []
        
        # Switch model to training mode
        self.model.eval()

        # Setup progress bar
        progress_bar = tqdm(iterator)
        for i, (img, lbl) in enumerate(progress_bar):
            # 1: Load sources, targets
            img = img.to(self.device)
            lbl = lbl.to(self.device)
    
            # 2: Get network outputs
            out = self.model(img)
            
            # 3: Calculate the loss
            loss = self.loss(out, lbl)
            
            # 4: Update loss
            total_loss.append(loss.item())
            
            # 5: Update metric
            out = out.detach()
            lbl = lbl.detach()

            acc = (out.argmax(dim=1) == lbl).float().mean()
            total_acc.append(acc.item())

        print("++++++++++++++ Evaluation result ++++++++++++++")
        loss = sum(total_loss) / len(iterator)
        print('Loss: ', loss)
        accuracy = sum(total_acc) / len(iterator)
        print('Accuracy: ', accuracy)
        # Upload tensorboard
        self.tsboard.update_loss('val', loss, epoch)
        self.tsboard.update_metric('val', accuracy, epoch)
        return loss, accuracy

    def train(self):
        start = time.time()
        for epoch in range(self.nepochs + 1):
            print('\nEpoch {:>3d}'.format(epoch))
            print('-----------------------------------')

            for i, group in enumerate(self.optimizer.param_groups):
                self.tsboard.update_lr(i, group['lr'], epoch)

            # Train phase
            train_loss = self.train_epoch(epoch, self.train_loader)
            
            # Eval phase
            val_loss, val_acc = self.val_epoch(epoch, self.val_loader)
            
            if epoch > self.config['optimizer']['warmup']:
                self.scheduler.step(val_loss)
                        
            if (epoch + 1) % self.val_step == 0:
                # Save weights
                self._save_model(epoch, train_loss, val_loss, val_acc)
        end = time.time()
        elapsed = str(timedelta(seconds=end-start))
        print(f'The model is trained in {elapsed} with {self.nepochs} epochs.')

    def _save_model(self, epoch, train_loss, val_loss, val_acc):
        
        data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(), 
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }

        loss_save_format = 'val_loss={val_loss:<.3}.pth'
        loss_save_format_filename = loss_save_format.format(
            val_loss=val_loss
        )
        
        metric_save_format = 'val_acc={val_acc:<.3}.pth'
        metric_save_format_filename = metric_save_format.format(
            val_acc=val_acc
        )

        if val_loss < self.best_loss:
            print(
                f'Loss is improved from {self.best_loss: .6f} to {val_loss: .6f}. Saving weights ...'
            )
            torch.save(data, os.path.join(self.save_dir, 'best_loss-' + loss_save_format_filename))
            self.best_loss = val_loss
        else:
            print(f'Loss is not improved from {self.best_loss: .6f}.')
        
        if val_acc > self.best_metric:
            print(
                f'Accuracy is improved from {self.best_metric: .6f} to {val_acc: .6f}. Saving weights ...'
            )
            torch.save(data, os.path.join(self.save_dir, 'best_acc-' + metric_save_format_filename))
            self.best_metric = val_acc
        else:
            print(f'Accuracy is not improved from {self.best_metric: .6f}.')

        

