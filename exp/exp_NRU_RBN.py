from data.data_loader import Dataset_ETT_hour,Dataset_ETT_day
from exp.exp_basic import Exp_Basic
from models.model import NRU_RBN
from utils_NRU_RBN.tools import EarlyStopping, adjust_learning_rate
from utils_NRU_RBN.metrics import metric
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import time
from utils_NRU_RBN.common_utils import commonmetric
from utils_NRU_RBN.tools import loss_fn_sigma
from  utils_NRU_RBN.crps import crps_gaussian
import warnings
warnings.filterwarnings('ignore')

class Exp_NRU_RBN(Exp_Basic):
    def __init__(self, args):
        super(Exp_NRU_RBN, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'NRU_RBN':NRU_RBN,

        }
        if self.args.model=='NRU_RBN':
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.d_model, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.likeloss,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'exchange':Dataset_ETT_day,
            'weather':Dataset_ETT_hour,
            'electrans':Dataset_ETT_hour,
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag =='val':
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss() 
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        crps_total=[]
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(tqdm(vali_loader)):
            pred, true,sigmaout,correlation,correlationvariance,res= self._process_one_batch(vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss=loss_fn_sigma(pred, sigmaout, true,correlation,correlationvariance,self.args.lossregularizer)
            true=true[:,-self.args.pred_len:,-1:]
            pred=pred[:,-self.args.pred_len:,-1:]
            sigmaout=sigmaout[:,-self.args.pred_len:,-1:]
            crps_total.append(crps_gaussian(true.detach().cpu(),pred.detach().cpu(),sigmaout.detach().cpu()))
            total_loss.append(loss.item())
        total_loss = np.mean(total_loss)
        self.model.train()
        return total_loss,crps_total

    def retdataloader(self,flag):
        return self._get_data(flag = 'train')

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')
        
        path = self.args.checkpoints
        best_model_path = os.path.join(path,setting+'.pth')
        if os.path.exists(best_model_path):
            print("load:",setting)
            self.model.load_state_dict(torch.load(best_model_path))
        else:
             print("No File, Train new")
           
        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(tqdm(train_loader)):
                iter_count += 1
                model_optim.zero_grad()
                
                pred, true,sigmaout,correlation,correlationvariance,res = self._process_one_batch(train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss=loss_fn_sigma(pred, sigmaout, true,correlation,correlationvariance,self.args.lossregularizer)
                train_loss.append(loss.item())
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
    
                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss,crps_total = self.vali(vali_data, vali_loader, criterion)
            print("vali_CRPS_mean:"+str(np.mean(np.array(crps_total))),"vali_CRPS_var:"+str(np.var(np.array(crps_total))))
            test_loss,crps_total = self.vali(test_data, test_loader, criterion)
            print("test_CRPS_mean:"+str(np.mean(np.array(crps_total))),"test_CRPS_var:"+str(np.var(np.array(crps_total))))
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, best_model_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        #best_model_path = path+setting+'_checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting,evaluate=False):
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()
        
        preds = []
        trues = []
        ress=[]
        sigmaouts = []
        crps_total=[]
        path = self.args.checkpoints
        best_model_path = os.path.join(path,setting+'.pth')
        if evaluate:
            if os.path.exists(best_model_path):
                print("load:",setting)
                self.model.load_state_dict(torch.load(best_model_path))
            else:
                print("No File")

        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(tqdm(test_loader)):
            self
            pred, true,sigmaout,_,_,res= self._process_one_batch(
                    test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            true=true[:,-self.args.pred_len:,-1:]
            pred=pred[:,-self.args.pred_len:,-1:]
            sigmaout=sigmaout[:,-self.args.pred_len:,-1:]
            res=res[:,-self.args.pred_len:,-1:]
            crps_total.append(crps_gaussian(true.detach().cpu(),(pred).detach().cpu(),sigmaout.detach().cpu()))
            sigmaouts.append(sigmaout.detach().cpu().numpy())
            preds.append((pred).detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            ress.append(res.detach().cpu().numpy())
        crpsret=crps_total
        print("test_CRPS_mean:"+str(np.mean(np.array(crps_total))),"test_CRPS_var:"+str(np.var(np.array(crps_total))))
        preds = np.array(preds)
        trues = np.array(trues)
        ress = np.array(ress)
        sigmaouts = np.array(sigmaouts)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        ress = ress.reshape(-1, ress.shape[-2], ress.shape[-1])
        sigmaouts = sigmaouts.reshape(-1, sigmaouts.shape[-2],sigmaouts.shape[-1])
        x,y,z=trues.shape
        for plot_id, i in enumerate(np.random.choice(range(len(true)), size=4, replace=False)):
            subplots = [221, 222, 223, 224]
            ff, yy,ss,rs,tr = preds[i], trues[i],sigmaouts[i],ress[i],preds[i]-ress[i]
            plt.subplot(subplots[plot_id])
            plt.grid()
            #plot_scatter(range(0, backcast_length), xx, color='b')
            ff=ff.squeeze()
            ss=ss.squeeze()
            plt.fill_between(range(0,y),ff - 2 * ss,
                         ff + 2 * ss, color='blue',
                         alpha=0.2)
            plt.plot(range(0,y), yy, color='g')
            #plt.scatter(range(0,y), yy, color='g')
            plt.plot(range(0,y),ff, color='r')
            #plt.scatter(range(0,y),ff, color='r')
            plt.plot(range(0,y),rs, color='b')
            #plt.scatter(range(0,y),rs, color='b')
            plt.plot(range(0,y),tr, color='y')
            #plt.scatter(range(0,y),tr, color='y')
        plt.show()

        # result save
        folder_path = self.args.root_path+'/results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)
        sqtrue=trues.squeeze()
        sqpred=preds.squeeze()
        resultout = commonmetric(sqpred,sqtrue)
        print(resultout)
        return resultout,0,0,crpsret

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
      batch_x = batch_x.float().to(self.device)
      batch_y = batch_y.float()
      batch_x_mark = batch_x_mark.float().to(self.device)
      batch_y_mark = batch_y_mark.float().to(self.device)
      
      # decoder input
      if self.args.padding==0:
          dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
      elif self.args.padding==1:
          dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
      dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
      # encoder - decoder
      if self.args.use_amp:
          with torch.cuda.amp.autocast():
              outputs,sigmaout,correlation,correlationvariance,ressample= self.model(batch_x, batch_x_mark,dec_inp, batch_y_mark)
      else:
        outputs,sigmaout,correlation,correlationvariance,ressample= self.model(batch_x, batch_x_mark,dec_inp, batch_y_mark)
      if self.args.inverse:
          outputs = dataset_object.inverse_transform(outputs)
      f_dim = -1 if self.args.features=='MS' else 0
      #batch_y = batch_y[:,-self.args.pred_len:,f_dim:]
      batch_y = batch_y.to(self.device)
      return outputs, batch_y,sigmaout , correlation,correlationvariance,ressample

        
