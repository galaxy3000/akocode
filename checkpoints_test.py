import torch
from tqdm import tqdm
import torch.nn as nn
from model.Model import Model
from utils.dataloader import SVFNDDataset
from utils.metrics import *
from utils.tools import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

path_fakesv = 'best_test_epoch10_0.8727'
path_fakett = 'best_test_epoch9_0.8528'


def load_chechpoint(path,dataset):
    model_path = ''
    if dataset == 'fakesv':
        token = pretrain_bert_wwm_token()
        from utils.dataloader import fakesv_collate_fn as collate_fn
    else:
        token = pretrain_bert_uncased_token()
        from utils.dataloader import fakett_collate_fn as collate_fn
    checkpoint_path = 'check_points/' + dataset + '/' + 'Model' + '/' + path
    model = Model(dataset=dataset, model_path=model_path, module_deep=1, n_tokens=16, 
                               num_heads=8, trans_dim=512, dropout=0.1)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    return model.cuda()


def get_dataloader(dataset):
    if dataset == 'fakesv':
        token = pretrain_bert_wwm_token()
        from utils.dataloader import fakesv_collate_fn as collate_fn
    else:
        token = pretrain_bert_uncased_token()
        from utils.dataloader import fakett_collate_fn as collate_fn
    dataset_test = SVFNDDataset('vid_time3_test.txt',token,dataset)
    test_dataloader = DataLoader(dataset_test, batch_size=16,
                                 num_workers=0,
                                 pin_memory=True,
                                 shuffle=False,
                                 # worker_init_fn=_init_fn,
                                 collate_fn=collate_fn)
    return test_dataloader


def test():
    dataset = 'fakett'
    if dataset == 'fakesv':
        model = load_chechpoint(path_fakesv,dataset)
    else:
        model = load_chechpoint(path_fakett,dataset)
    test_dataloader = get_dataloader(dataset)
    tpred = []
    tlabel = []
    for batch in tqdm(test_dataloader):
        batch_data = batch
        # 把每个样本都放到gpu上
        for k, v in batch_data.items():
            batch_data[k] = v.cuda()
        label = batch_data['label']
        with torch.set_grad_enabled(False):
            outputs = model(**batch_data)
            _,preds = torch.max(outputs, 1)
        tlabel.extend(label.detach().cpu().numpy().tolist())
        tpred.extend(preds.detach().cpu().numpy().tolist())
    results = metrics(tlabel, tpred)
    get_confusionmatrix_fnd(tpred,tlabel)
    print(results)



if __name__ == '__main__':
    test()
