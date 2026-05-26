import copy
import os
import time
import numpy as np
import tqdm
from tqdm import tqdm
from utils.metrics import *
import torch
from torch import nn
from collections import defaultdict  
import torch.nn.functional as F


class Trainer():
    def __init__(self,
                model,
                 auxiliary_w,
                 device,
                 lr,
                 dropout,
                 dataloaders,
                 weight_decay,
                 save_param_path,
                 writer, 
                 epoch_stop,
                 epoches,
                 save_threshold = 0.0, 
                 start_epoch = 0,
                 ):
        
        self.model = model
        self.alpha = auxiliary_w
        self.device = device
        self.dataloaders = dataloaders
        self.start_epoch = start_epoch
        self.num_epochs = epoches
        self.epoch_stop = epoch_stop
        self.save_threshold = save_threshold
        self.writer = writer
        
        
        if os.path.exists(save_param_path):
            self.save_param_path = save_param_path
        else:
            self.save_param_path = os.makedirs(save_param_path)
            self.save_param_path= save_param_path

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
    
        self.criterion = nn.CrossEntropyLoss()
                    
    def train(self):
        since = time.time()
        self.model.cuda()
        best_acc_val = 0.0
        best_epoch_val = 0
        is_earlystop = False
        last_save_path = ''

        for epoch in range(self.start_epoch, self.start_epoch+self.num_epochs):
            if is_earlystop:
                break
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch+1, self.start_epoch+self.num_epochs))
            print('-' * 50)

            # updata lr
            p = float(epoch) / 100
            lr = self.lr / (1. + 10 * p) ** 0.75
            
            # 创建优化器
            # self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
            
            # 分离参数
            spatial_params = [] # for modality_weights
            main_params = []  

            for name, param in self.model.named_parameters():
                if 'modality_weights_raw' in name:
                    spatial_params.append(param)
                    # print(f"Found modality_weights: {name}, adding to specialized group.")
                else:
                    main_params.append(param)

            self.optimizer = torch.optim.AdamW([
                {'params': main_params, 'lr': lr},
                {'params': spatial_params, 'lr': lr * 100, 'weight_decay': 0.0}
            ], weight_decay=1e-2)

            # print(f"Optimizer initialized with base_lr={lr} and spatial_lr={lr*100}")
            
            for phase in ['train', 'val', 'test']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                print('-' * 10)
                print (phase.upper())
                print('-' * 10)

                running_loss = 0.0 
                
                tpred = []
                tlabel = []
                    
                for batch in tqdm(self.dataloaders[phase]):
                    batch_data=batch
                    # to gpu
                    for k,v in batch_data.items():
                        batch_data[k]=v.cuda()
                    label = batch_data['label']

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(**batch_data)
                        _, preds = torch.max(outputs, 1)
                        
                        loss = self.criterion(outputs, label)
                        # total_loss = loss + self.alpha * auxiliary_loss
                        # total_loss = loss

                        if phase == 'train':
                            self.optimizer.zero_grad()
                            loss.backward()
                            # self._apply_gradient_scales(scales)
                            # grads = self.model.UnifiedTransformer.attentions[0].modality_weights_raw.grad
                            # print(f"Gradients of modality_weights: {grads}")
                            self.optimizer.step()

                    tlabel.extend(label.detach().cpu().numpy().tolist())
                    tpred.extend(preds.detach().cpu().numpy().tolist())
                    
                    running_loss += loss.item() * label.size(0)
                
                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                # losses = [loss,auxiliary_loss]
                
                # final result
                print('Loss: {:.4f} '.format(epoch_loss))
                final_results = metrics(tlabel, tpred)
                print(final_results)
                get_confusionmatrix_fnd(tpred,tlabel)
                
                if phase == 'test' and final_results['acc'] > best_acc_val:
                    best_acc_val = final_results['acc']
                    best_epoch_val = epoch + 1
                    if best_acc_val > self.save_threshold:
                        if os.path.exists(last_save_path):
                            print('delete the previous checkpoint...')
                            os.remove(last_save_path)
                        save_path = self.save_param_path + "_test_epoch" + str(best_epoch_val) + "_{0:.4f}".format(best_acc_val)
                        torch.save(self.model.state_dict(),save_path)
                        last_save_path = save_path
                        print("saved " + self.save_param_path + "_test_epoch" + str(
                            best_epoch_val) + "_{0:.4f}".format(best_acc_val))
                    else:
                        if epoch - best_epoch_val >= self.epoch_stop - 1:
                            is_earlystop = True
                            print("early stopping...")

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print("Best model on val: epoch" + str(best_epoch_val) + "_" + str(best_acc_val))
        return True
