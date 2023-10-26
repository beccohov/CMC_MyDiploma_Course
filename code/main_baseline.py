import os
os.system('mkdir wheelhouse')
os.system('echo \'python_gdcm\' > requirements.txt')
os.system('cat requirements.txt')
os.system('pip download -r requirements.txt -d wheelhouse')
os.system('echo \'torch-scatter\' > requirements.txt')
os.system('pip download -r requirements.txt -d wheelhouse')
os.system('echo 'torch-geometric' > requirements.txt')
os.system('pip download -r requirements.txt -d wheelhouse')
os.system('echo 'pylibjpeg' > requirements.txt')
os.system('pip download -r requirements.txt -d wheelhouse')
os.system('pip install torch-geometric')

import timm
from timm.scheduler import  CosineLRScheduler
import numpy as np
import pandas as pd
import os
from tqdm import tqdm 

import sklearn,sklearn.model_selection
import torch
from torch import nn
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch.utils.data import DataLoader, Dataset
#from timm.scheduler import CosineLRScheduler
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def load_df(directory):
    splits = ["test"]
    dfs = dict()
    
    for split in splits:
        path = os.path.join(directory, split)
        files = os.listdir(path)
        list_df = []
        
        for file in files:
            d = dict(np.load(os.path.join(path,file)))
            d['file'] = file
            list_df.append(d)
            
        dfs[split] = pd.DataFrame.from_dict(list_df)
        
    return dfs

layout_xla_random_test = load_df(npz_all/npz/layout/xla/random/")
layout_xla_default_test = load_df("npz_all/npz/layout/xla/default/")
layout_nlp_random_test = load_df("npz_all/npz/layout/nlp/random/")
layout_nlp_default_test = load_df("npz_all/npz/layout/nlp/default/")
def load_df_train_name(directory):
    splits = ["train", "valid"]
    dfs = dict()
    
    for split in splits:
        path = os.path.join(directory, split)
        files = os.listdir(path)
        list_df = []
        
        for file in files:                         
            list_df.append(os.path.join(path,file))
        dfs[split] = pd.DataFrame.from_dict(list_df)
        
    return dfs

layout_xla_random = load_df_train_name("npz_all/npz/layout/xla/random/")
layout_xla_default = load_df_train_name("npz_all/npz/layout/xla/default/")

layout_nlp_random = load_df_train_name("npz_all/npz/layout/nlp/random/")
layout_nlp_default = load_df_train_name("npz_all/npz/layout/nlp/default/")

class TileDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx][0]
        config_feat = torch.tensor(row['node_config_feat'].astype(np.float32))
        node_feat = torch.tensor(row['node_feat'].astype(np.float32))
        node_opcode = torch.tensor(row['node_opcode'].astype(np.int32))
        edge_index = torch.tensor(np.swapaxes(row['edge_index'],0,1).astype(np.int32))
        target = (row['config_runtime']).astype(np.float32)
        # minmax scale the target, we only care about order
        target = (target-min(target))/(max(target) -min(target))
        target = torch.tensor(target)
        return config_feat,node_feat,node_opcode,edge_index,target
    

class TileDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name = self.df.iloc[idx]
        return name
    
dataset = TileDataset(layout_xla_default["train"])

class SimpleModel(torch.nn.Module):
    def __init__(self, hidden_channels, graph_feats, hidden_dim):
        super().__init__()
        op_embedding_dim = 4 # I choose 4-dimensional embedding
        self.embedding = torch.nn.Embedding(120, #120 different op-codes
                                            op_embedding_dim,
                                           )
        assert len(hidden_channels)>0
        in_channels = op_embedding_dim+140
        self.convs = torch.nn.ModuleList()
        last_dim = hidden_channels[0]
        self.convs.append(GCNConv(in_channels, hidden_channels[0]))
        for i in range(len(hidden_channels)-1):
            self.convs.append(GCNConv(hidden_channels[i], hidden_channels[i+1]))
            last_dim = hidden_channels[i+1]
        self.convs.append(GCNConv(last_dim, graph_feats))
        
        self.dense = torch.nn.Sequential(nn.Linear(82, 64),
                                         nn.ReLU(),
                                         nn.Linear(64, 64),
                                         nn.ReLU(),
                                         nn.Linear(64, 1),
                                        )
    
    def forward(self, x_cfg: Tensor,x_feat: Tensor, x_op: Tensor, edge_index: Tensor) -> Tensor:
        
        #get graph features
        x_cfg = x_cfg.mean(dim=1)
        #print(x_cfg.shape)
        x = torch.concat([x_feat,self.embedding(x_op)],dim = 1)
        #pass though conv layers
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        # get 1d graph embedding using average pooling
        x_graph = torch.mean(x,0)
        
        
        #put graph data into config data
        x = torch.concat([x_cfg,x_graph.repeat((len(x_cfg),1))],axis=1) #torch.Size([10528, 225])
        #put into dense nn
        #print(x.shape)
        x = torch.flatten(self.dense(x))
        return x

model = SimpleModel(hidden_channels = [16,32,16,48],graph_feats = 64,hidden_dim=64).to(device)
df = pd.concat((layout_xla_default["train"],layout_xla_default["valid"]),axis=0).reset_index(drop=True)
df = pd.concat((df,layout_xla_random["valid"]),axis=0).reset_index(drop=True)
df = pd.concat((df,layout_xla_random["train"]),axis=0).reset_index(drop=True)

df = pd.concat((df,layout_nlp_default["train"]),axis=0).reset_index(drop=True)
df = pd.concat((df,layout_nlp_default["valid"]),axis=0).reset_index(drop=True)

df = pd.concat((df,layout_nlp_random["train"]),axis=0).reset_index(drop=True)
df = pd.concat((df,layout_nlp_random["valid"]),axis=0).reset_index(drop=True)

kfold = sklearn.model_selection.KFold(n_splits=5,shuffle=True,random_state=0)
score_means = []
score_maxs = []
for fold,(tr_idx,va_idx) in enumerate(kfold.split(df)):
    if(fold>=1):
        continue
    model = SimpleModel(hidden_channels = [16,32,16,48],graph_feats = 64,hidden_dim=64).to(device)
    train_dataset = TileDataset(df.iloc[tr_idx])
    val_dataset = TileDataset(df.iloc[va_idx])
    criterion = torch.nn.MSELoss()
    steps = len(train_dataset)*10
    warmup_steps = int(steps*0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay = 1e-5)
    scheduler = CosineLRScheduler(optimizer,t_initial= steps,warmup_t=warmup_steps, warmup_lr_init=1e-6,lr_min=2e-8,)
    
    def score_tile_mean(predictions, df):
        score = 0
        for i in range(len(df)):
            predbest = np.mean(df.iloc[i]['config_runtime'][predictions[i]])
            best = np.mean(np.sort(df.iloc[i]['config_runtime'])[:5])
            score += 2-predbest/best
        score /= len(df)
        return score
    def score_tile_max(predictions, df):
        score = 0
        for i in range(len(df)):
            predbest = np.min(df.iloc[i]['config_runtime'][predictions[i]])
            best = np.min(df.iloc[i]['config_runtime'])
    #         print(best,predbest)
            score += 2 - predbest/best
        score /= len(df)
        return score

    best_score = 0
    best_score_max = 0
    score_best = 10000000

    for epoch in range(20):
        model.train()
        pbar = tqdm(range(len(train_dataset)),leave=False)
        loss_sum = 0
        n = 0
        for i in pbar:
            d = dict(np.load(str(train_dataset[i][0])))
            
            config_feat = torch.tensor(d['node_config_feat'].astype(np.float32))
            node_feat = torch.tensor(d['node_feat'].astype(np.float32))
            node_opcode = torch.tensor(d['node_opcode'].astype(np.int32))
            edge_index = torch.tensor(np.swapaxes(d['edge_index'],0,1).astype(np.int32))
            target = (d['config_runtime']).astype(np.float32)
            # minmax scale the target, we only care about order
            target = (target-min(target))/(max(target) -min(target))
            target = torch.tensor(target)
            
            #cfg_ft,nd_ft,nd_op,ind,target = train_dataset[i]
            cfg_ft,nd_ft,nd_op,ind,target = config_feat.to(device),node_feat.to(device),node_opcode.to(device),edge_index.to(device),target.to(device)

            out = model(cfg_ft,nd_ft,nd_op,ind)
            
            #break
            loss = criterion(out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-2)
            scheduler.step(i+len(train_dataset)*epoch)
            optimizer.step()
            loss_sum+=loss.item()
            n+=1
            pbar.set_description(f'running loss: {(loss_sum/n):.2f},current loss: {(loss.item()):.2f}')
        #break
        pbar.close()
        model.eval()
        tile_xla_predictions = []
        score_now = 0
        pbar = tqdm(range(len(val_dataset)),leave=False)
        for i in pbar:
            
            d = dict(np.load(str(val_dataset[i][0])))
            
            config_feat = torch.tensor(d['node_config_feat'].astype(np.float32))
            node_feat = torch.tensor(d['node_feat'].astype(np.float32))
            node_opcode = torch.tensor(d['node_opcode'].astype(np.int32))
            edge_index = torch.tensor(np.swapaxes(d['edge_index'],0,1).astype(np.int32))
            target = (d['config_runtime']).astype(np.float32)
            # minmax scale the target, we only care about order
            target = (target-min(target))/(max(target) -min(target))
            target = torch.tensor(target)
            
            #cfg_ft,nd_ft,nd_op,ind,target = val_dataset[i]
            cfg_ft,nd_ft,nd_op,ind,target = config_feat.to(device),node_feat.to(device),node_opcode.to(device),edge_index.to(device),target.to(device)

            out = model(cfg_ft,nd_ft,nd_op,ind)
            score_now += criterion(out, target)
            tile_xla_predictions.append(np.argsort(out.cpu().detach().numpy())[:5])
        pbar.close()
        #score_mean = score_tile_mean(tile_xla_predictions, val_dataset.df)
        #score_max = score_tile_max(tile_xla_predictions, val_dataset.df)
        print(f'fold {fold} epoch {epoch}, best = {score_best:.3f}, now = {score_now:.3f},')
        if score_best > score_now:
            score_best = score_now
        #best_score_max = score_max
            torch.save(model.state_dict(), f'layout_xla_default_best_model_{fold}.pth')
    #score_means.append(best_score)
    #score_maxs.append(best_score_max)
#print(f'comp_score = {np.mean(score_maxs)}, mean_score = {np.mean(score_means)},')
dataset = TileDataset(layout_xla_default_test["test"])
tile_xla_predictions = [[] for i in range(len(dataset))]
for fold in range(1):
    model.load_state_dict(torch.load(f'layout_xla_default_best_model_{fold}.pth'))
    model.eval()
    pbar = tqdm(range(len(dataset)))
    for i in pbar:
        cfg_ft,nd_ft,nd_op,ind,target = dataset[i]
        cfg_ft,nd_ft,nd_op,ind,target = cfg_ft.to(device),nd_ft.to(device),nd_op.to(device),ind.to(device),target.to(device)

        out = model(cfg_ft,nd_ft,nd_op,ind)
        tile_xla_predictions[i].append(out.cpu().detach().numpy())
tile_xla_predictions = [np.argsort(np.mean(pred,axis=0))[:-1] for pred in tile_xla_predictions]
tile_xla_predictions[0]
sub = pd.read_csv('sample_submission.csv')
for i,filename in enumerate(layout_xla_random_test["test"]['file'].values):
    id = 'layout:xla:default:' +filename[:-4]
    print(id)
    sub.loc[sub.ID == id,'TopConfigs'] = ';'.join(tile_xla_predictions[i].astype(str))
sub.to_csv('submission.csv',index=False)
sub
dataset = TileDataset(layout_xla_random_test["test"])
tile_xla_predictions = [[] for i in range(len(dataset))]
for fold in range(1):
    model.load_state_dict(torch.load(f'layout_xla_default_best_model_{fold}.pth'))
    model.eval()
    pbar = tqdm(range(len(dataset)))
    for i in pbar:
        cfg_ft,nd_ft,nd_op,ind,target = dataset[i]
        cfg_ft,nd_ft,nd_op,ind,target = cfg_ft.to(device),nd_ft.to(device),nd_op.to(device),ind.to(device),target.to(device)

        out = model(cfg_ft,nd_ft,nd_op,ind)
        tile_xla_predictions[i].append(out.cpu().detach().numpy())
tile_xla_predictions = [np.argsort(np.mean(pred,axis=0))[:-1] for pred in tile_xla_predictions]
tile_xla_predictions[0]

#sub = pd.read_csv('/kaggle/input/predict-ai-model-runtime/sample_submission.csv')
for i,filename in enumerate(layout_xla_random_test["test"]['file'].values):
    id = 'layout:xla:random:' +filename[:-4]
    print(id)
    sub.loc[sub.ID == id,'TopConfigs'] = ';'.join(tile_xla_predictions[i].astype(str))
sub.to_csv('submission.csv',index=False)
sub
dataset = TileDataset(layout_nlp_random_test["test"])
tile_xla_predictions = [[] for i in range(len(dataset))]
for fold in range(1):
    model.load_state_dict(torch.load(f'layout_xla_default_best_model_{fold}.pth'))
    model.eval()
    pbar = tqdm(range(len(dataset)))
    for i in pbar:
        cfg_ft,nd_ft,nd_op,ind,target = dataset[i]
        cfg_ft,nd_ft,nd_op,ind,target = cfg_ft.to(device),nd_ft.to(device),nd_op.to(device),ind.to(device),target.to(device)

        out = model(cfg_ft,nd_ft,nd_op,ind)
        tile_xla_predictions[i].append(out.cpu().detach().numpy())
tile_xla_predictions = [np.argsort(np.mean(pred,axis=0))[:-1] for pred in tile_xla_predictions]
tile_xla_predictions[0]

#sub = pd.read_csv('/kaggle/input/predict-ai-model-runtime/sample_submission.csv')
for i,filename in enumerate(layout_nlp_random_test["test"]['file'].values):
    id = 'layout:nlp:random:' +filename[:-4]
    print(id)
    sub.loc[sub.ID == id,'TopConfigs'] = ';'.join(tile_xla_predictions[i].astype(str))

dataset = TileDataset(layout_nlp_default_test["test"])
tile_xla_predictions = [[] for i in range(len(dataset))]
for fold in range(1):
    model.load_state_dict(torch.load(f'layout_xla_default_best_model_{fold}.pth'))
    model.eval()
    pbar = tqdm(range(len(dataset)))
    for i in pbar:
        cfg_ft,nd_ft,nd_op,ind,target = dataset[i]
        cfg_ft,nd_ft,nd_op,ind,target = cfg_ft.to(device),nd_ft.to(device),nd_op.to(device),ind.to(device),target.to(device)

        out = model(cfg_ft,nd_ft,nd_op,ind)
        tile_xla_predictions[i].append(out.cpu().detach().numpy())
tile_xla_predictions = [np.argsort(np.mean(pred,axis=0))[:-1] for pred in tile_xla_predictions]
tile_xla_predictions[0]

for i,filename in enumerate(layout_nlp_default_test["test"]['file'].values):
    id = 'layout:nlp:default:' +filename[:-4]
    print(id)
    sub.loc[sub.ID == id,'TopConfigs'] = ';'.join(tile_xla_predictions[i].astype(str))
sub.to_csv('result.csv',index=False)
