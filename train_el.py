import torch 
import torch.nn as nn
import numpy as np
import gc
import pandas as pd
from rdflib import Graph,Namespace
import json 
import sys 
import csv

def getIndex(e,mapping):
    if(e.split('/')[-1].split('#')[-1] in mapping.keys()):
        return mapping.get(e.split('/')[-1].split('#')[-1]),mapping
    if(e.split('/')[-1].split('#')[-1] not in mapping.keys()):
        mapping[e.split('/')[-1].split('#')[-1]]=len(mapping.keys())
        return mapping.get(e.split('/')[-1].split('#')[-1]),mapping

def embed(mapping_e, mapping_p, graph,max_e,max_p):
    tensor = np.zeros((max_e,max_e,max_p), dtype=np.int32)
    for s,p,o in graph.triples((None,None,None)):
        idx_s,mapping_e=getIndex(s,mapping_e)
        idx_o,mapping_e=getIndex(o,mapping_e)
        idx_p,mapping_p=getIndex(p,mapping_p)
       
        tensor[idx_s][idx_o][idx_p]=1
    return tensor, mapping_e,mapping_p

class CustomEvaluation():
    def __init__(self,mapping_e,mapping_p,max_e,max_p,batch_size):
        self.accuracy_path="/home/ubuntu/data/home/ubuntu/deeplogic/metrics/el/validation/accuracy.csv"
        self.tp_path="/home/ubuntu/data/home/ubuntu/deeplogic/metrics/el/validation/tp.csv"
        self.tn_path="/home/ubuntu/data/home/ubuntu/deeplogic/metrics/el/validation/tn.csv"
        self.fp_path="/home/ubuntu/data/home/ubuntu/deeplogic/metrics/el/validation/fp.csv"
        self.fn_path="/home/ubuntu/data/home/ubuntu/deeplogic/metrics/el/validation/fn.csv"
        self.loss_path="/home/ubuntu/data/home/ubuntu/deeplogic/metrics/el/validation/loss.csv"
        self.recall_path="/home/ubuntu/data/home/ubuntu/deeplogic/metrics/el/validation/recall.csv"
        self.precision_path="/home/ubuntu/data/home/ubuntu/deeplogic/metrics/el/validation/precision.csv"
        self.soundness_path="/home/ubuntu/data/home/ubuntu/deeplogic/metrics/el/validation/soundness.csv"
        self.quality_path="/home/ubuntu/data/home/ubuntu/deeplogic/metrics/el/validation/quality.csv"
        self.f1_path="/home/ubuntu/data/home/ubuntu/deeplogic/metrics/el/validation/f1.csv"
        self.absolute_path="/home/ubuntu/data/home/ubuntu/deeplogic/metrics/el/validation/"
        self.tot = max_e*max_e*max_p*batch_size
        self.max_e=max_e
        self.max_p=max_p
        self.batch_size=batch_size
        self.mapping_p=mapping_p
        self.mapping_e=mapping_e
            
        

    def update(self,y_pred_batch,y_true_batch):

        corrects= y_pred_batch.eq(y_true_batch.view_as(y_pred_batch)).sum().item()
        
        for i in range(self.batch_size):
            accuracy=corrects/self.tot
            print(accuracy,flush=True)
            self.write_csv(self.accuracy_path,accuracy)
            y_true = y_true_batch[0][i]
            y_pred = y_pred_batch[0][i]
            tn,fp,fn,tp = 0,0,0,0
            #tp = (((y_pred-y_true)>0) & ((y_pred-y_true)<1)).sum().item()
            #tn = ((y_pred-y_true)==0).sum().item()
            #fp = ((y_pred-y_true)==1).sum().item()
            #fn = ((y_pred-y_true)<0).sum().item()
            for row in range(self.max_e):
                for col in range(self.max_e):
                    for sli in range(self.max_p):
                        #print(y_pred[row][col][sli].item(),flush=True)
                        #print(y_true[row][col][sli].item(),flush=True)
                        yp=y_pred[row][col][sli].item()
                        yt=y_true[row][col][sli].item()*0.5
                        typ=yp-yt
                        #print(typ,flush=True)
                        if(str(typ)=='0.0'):
                            tn=tn+1
                        if(typ>0.9):
                            fp=fp+1
                        if(typ<0):
                            fn=fn+1
                        if((typ>0) and (typ<1)):
                            tp=tp+1
            if(tn==0):
                print("ntro tn=0",flush=True)
                tn=1
            if(tp==0):
                print("ntro tp=0",flush=True)
                tp=1
            if(fp==0):
                print("ntro fp=0",flush=True)
                fp=1
            if(fn==0):
                print("ntro fn=0",flush=True)
                fn=1
            self.write_csv(self.tp_path,tp)
            self.write_csv(self.fp_path,fp)
            self.write_csv(self.tn_path,tn)
            self.write_csv(self.fn_path,fn)
            for prop in list(self.mapping_p.keys()):
                p = self.mapping_p[prop]
                val=y_pred[:][:][p].eq(y_true[:][:][p].view_as(y_pred[:][:][p])).sum().item()/(self.max_e*self.max_e)
                self.write_csv(self.absolute_path+str(prop)+".csv",val)
            print("tp:"+str(tp)+",fp:"+str(fp)+",tn:"+str(tn)+",fn:"+str(fn),flush=True)
            precision=tp/(tp+fp)
            print("precisio",flush=True)
            self.write_csv(self.precision_path,precision)
            recall=tp/(tp+fn)
            print("recall",flush=True)
            print(precision,flush=True)
            print(recall,flush=True)
            self.write_csv(self.recall_path,recall)
            f1=2*precision*recall/(precision+recall)
            self.write_csv(self.f1_path,f1)
            soundness=(tn+fn+tp)/(tn+fp+fn+tp)
            self.write_csv(self.soundness_path,soundness)
            quality=(tn+tp)/(tn+fp+fn+tp)
            self.write_csv(self.quality_path,quality)
       
    def update_loss(self,loss):
        self.write_csv(self.loss_path,loss)

    def write_csv(self,path,row):
        with open(path,'a') as fd:
            fd.write(str(row)+'\n')

    def update_dict(self,mapping_e,mapping_p):
        self.mapping_p = mapping_p
        self.mapping_e = mapping_e


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


def get_dataset(i,mapping_entities,mapping_properties,max_e,max_p):
    max_e,max_p=max_e,max_p
    X_train=[]
    Y_train=[]
    for j in range(i*500,(i+1)*500):
    
        n = Namespace("http://example.org/")
        g1 = Graph()
        g1.bind("ex",n)
        g1.parse("/home/ubuntu/data/home/ubuntu/deeplogic/train_pre/scene{}.ttl".format(j), format="turtle")
        spo_tensor,mapping_entities,mapping_properties = embed(mapping_entities,mapping_properties,g1,max_e,max_p)
        #counter = counter +1
        X_train.append(spo_tensor)
        #if(counter ==1000):
        #break
        #counter = 0
        #for filename in os.listdir("/home/ubuntu/data/home/ubuntu/deeplogic/el_dataset/y"):
        g2 = Graph()
        g2.bind("ex",n)
        g2.parse("/home/ubuntu/data/home/ubuntu/deeplogic/src/main/resources/dataset/dataset/train_materialized/el/scene{}.ttl".format(j),format="turtle")

        g3 = Graph()
        g3.parse("/home/ubuntu/data/home/ubuntu/deeplogic/clevr_el_materialized.ttl", format="ttl")

        graph = g2 - g3
        graph.bind("ex",n)

        spo_tensor,mapping_entities,mapping_properties = embed(mapping_entities,mapping_properties,graph,max_e,max_p)
        Y_train.append(spo_tensor)
        #counter = counter +1
        #if(counter ==1000):
        #break
    train_x = torch.from_numpy(np.array(X_train))
    train_y = torch.from_numpy(np.array(Y_train))
    

    batch_size = 5


    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(train_x,train_y)

    # data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)

    return train_loader,len(X_train),mapping_entities,mapping_properties

class AutoEncoder(nn.Module):
    def __init__(self,options,max_e,max_p,batch_size):
        super(AutoEncoder, self).__init__()
        self.down_layers=nn.ModuleList()
        self.up_layers=nn.ModuleList()
        self.down_pool_layers=nn.ModuleList()
        self.up_pool_layers=nn.ModuleList()
        self.n_layers = len(options)
        for i in range(self.n_layers):
            
          #Encoder
            kernel_size_wh = options[i][0]
            if(i==0): 
                self.down_layers.append(nn.Conv3d(batch_size,options[i][1],(kernel_size_wh,kernel_size_wh,1)))
            else:
                self.down_layers.append(nn.Conv3d(options[i-1][1],options[i][1],(kernel_size_wh,kernel_size_wh,1)))
            if(i!=(self.n_layers-1)):
                self.down_pool_layers.append(nn.MaxPool3d(kernel_size=2, stride=options[i][2]))
        for i in range(self.n_layers):
          #decoder
            if(i!=(self.n_layers-1)):
                self.up_layers.append(nn.ConvTranspose3d(options[self.n_layers-1-i][1],options[self.n_layers-1-i-1][1],(options[self.n_layers-1-i][0],options[self.n_layers-1-i][0],1)))
            else:
                #last layer
                self.up_layers.append(nn.ConvTranspose3d(options[0][1],batch_size,(options[0][0],options[0][0],1)))
            if(i!=(self.n_layers-1)):            
                self.up_pool_layers.append(nn.MaxUnpool3d(kernel_size=2, stride=options[self.n_layers-1-i][2]))
        for el in self.down_layers:
            torch.nn.init.normal_(el.weight, mean=0.5, std=0.7)
        for el in self.up_layers:
            torch.nn.init.normal_(el.weight, mean=0.5, std=0.7)

    def encode(self, x):
        # Encoder
        for i in range(len(self.down_layers)): 
            x = self.down_layers[i](x)
            x = torch.sigmoid(x)
        return x

    def forward(self, x):

        # Decoder
        x= self.encode(x)
        for i in range(len(self.up_layers)):
            x = self.up_layers[i](x)
            x = torch.sigmoid(x)
        
        return x

def train_model(options,num_e,num_p):
    print("start training",flush=True)
    max_e,max_p = num_e,num_p
    mapping_entities = {}
    mapping_properties = {}
    DEVICE = get_device()
    EPOCHS = 3
    BATCHSIZE=5
    criterion = torch.nn.BCELoss()
    evaluation = CustomEvaluation(mapping_entities,mapping_properties,max_e,max_p,BATCHSIZE)
    try: 
        model = AutoEncoder(options,max_e,max_p,BATCHSIZE).to(DEVICE)
        lr = 0.09
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        # Training of the model.
        for epoch in range(EPOCHS):
            print("Epoca"+str(epoch),flush=True)
            model.train()
            for i in range(100):
                train_loader,X_train_shape,mapping_entities,mapping_properties= get_dataset(i,mapping_entities,mapping_properties,max_e,max_p)
                evaluation.update_dict(mapping_entities,mapping_properties)
                N_TRAIN_EXAMPLES = X_train_shape
                for batch_idx, (data, target) in enumerate(train_loader):
                    # Limiting training data for faster epochs.
                    if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                        break
                    data, target = data[None, ...].to(DEVICE, dtype=torch.float), target[None, ...].to(DEVICE, dtype=torch.float)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    evaluation.update_loss(loss.data.item())
                    loss.backward()
                    optimizer.step()
                del train_loader
                gc.collect()
                
            # Validation of the model.
            model.eval()
            correct = 0
            tot = 0
            
            with torch.no_grad():
                for i in range(100,140):
                    valid_loader,X_valid_shape,mapping_entities,mapping_properties= get_dataset(i,mapping_entities,mapping_properties,max_e,max_p)
                    evaluation.update_dict(mapping_entities, mapping_properties)
                    N_VALID_EXAMPLES = X_valid_shape
                    for batch_idx, (data, target) in enumerate(valid_loader):
                        # Limiting validation data.
                        if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                            break
                        data, target = data[None, ...].to(DEVICE, dtype=torch.float), target[None, ...].to(DEVICE, dtype=torch.float)
                        output = model(data)
                        pred = torch.round(output)
                        newCorrect= pred.eq(target.view_as(pred)).sum().item()
                        correct += newCorrect
                        tot +=max_e*max_e*max_p*BATCHSIZE
                        evaluation.update(pred,target)
                    del valid_loader
                    gc.collect()
            accuracy = correct*100 / tot
            print('Epoch: {}  Loss: {}  Accuracy: {} %'.format(epoch, loss.data, accuracy),flush=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, '/home/ubuntu/data/home/ubuntu/deeplogic/el_model_params/eporch'+str(epoch))
        print("ACCURACY: " +str(accuracy),flush=True)
        with open('/home/ubuntu/data/home/ubuntu/deeplogic/el_model_params/mapping_e.json', 'w') as outfile:
            json.dump(mapping_entities, outfile)
        with open('/home/ubuntu/data/home/ubuntu/deeplogic/el_model_params/mapping_p.json', 'w') as outfile:
            json.dump(mapping_properties, outfile)
    except Exception as e:
        print(e)




if __name__ == "__main__":
    #el
    """
    {'n_layers': 4, 'kernel_size_l0': 9, 'stride_pool_0': 2, 'kernel_size_l1': 11, 'n_filter_l1': 2, 'stride_pool_1': 1, 'kernel_size_l2': 9, 'n_filter_l2': 3, 'stride_pool_2': 2, 'kernel_size_l3': 7, 'n_filter_l3': 4}
    """
    options=[[9,1,2],[11,2,1],[9,3,2],[7,4,1]]
    train_model(options,351,11)
    #ql
    """
     {'n_layers': 1, 'kernel_size_l0': 9}
    """
    #options=[[9,1,1]]
    #train_model(options,353,11)



