import optuna
from optuna.trial import TrialState
import torch 
import torch.nn as nn
import numpy as np
import gc
import random
from rdflib import Graph,Namespace
def getIndex(e,mapping):
    if(e.split('/')[-1].split('#')[-1] in mapping.keys()):
        return mapping.get(e.split('/')[-1].split('#')[-1]),mapping
    if(e.split('/')[-1].split('#')[-1] not in mapping.keys()):
        mapping[e.split('/')[-1].split('#')[-1]]=len(mapping.keys())
        return mapping.get(e.split('/')[-1].split('#')[-1]),mapping


def embed(mapping_e, mapping_p, graph,max_e,max_p):
    tensor = np.zeros((max_e,max_e,max_p), dtype=np.int32)
    midx_s,midx_o,midx_p=0,0,0
    n = Namespace("http://example.org/")
    q_n_blank = graph.query('SELECT ?s ?p ?o WHERE {?s ?p ?o filter(!isBlank(?s) && !isBlank(?o))}',initNs={ 'ex': n })
    #for s,p,o in graph.triples((None,None,None)):
    for row in q_n_blank:
        idx_s,mapping_e=getIndex(row.s,mapping_e)
        idx_o,mapping_e=getIndex(row.o,mapping_e)
        idx_p,mapping_p=getIndex(row.p,mapping_p)
        tensor[idx_s][idx_o][idx_p]=1
    return tensor, mapping_e,mapping_p


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


def get_dataset(i,index,mapping_entities,mapping_properties):
    X_train=[]
    Y_train=[]
    max_e,max_p = 353,11
    #counter = 0
    for j in range(i*500+index*5000,(i+1)*500+index*5000):
    #for filename in os.listdir("/home/ubuntu/data/home/ubuntu/deeplogic/el_dataset/x"):
        n = Namespace("http://example.org/")
        g1 = Graph()
        g1.bind("ex",n)
        g1.parse("/home/ubuntu/data/home/ubuntu/deeplogic/train_pre/scene{}.ttl".format(j), format="turtle")
        spo_tensor,mapping_entities,mapping_properties = embed(mapping_entities,mapping_properties,g1,max_e,max_p)
        #counter = counter +1
        X_train.append(spo_tensor)
        g2 = Graph()
        g2.bind("ex",n)
        g2.parse("/home/ubuntu/data/home/ubuntu/deeplogic/src/main/resources/dataset/dataset/train_materialized/ql/scene{}.ttl".format(j),format="turtle")

        g3 = Graph()
        g3.parse("/home/ubuntu/data/home/ubuntu/deeplogic/clevr_ql_materialized.ttl", format="ttl")

        graph = g2 - g3
        graph.bind("ex",n)

        spo_tensor,mapping_entities,mapping_properties = embed(mapping_entities,mapping_properties,graph,max_e,max_p)
        Y_train.append(spo_tensor)
    train_x = torch.from_numpy(np.array(X_train)).float()
    train_y = torch.from_numpy(np.array(Y_train)).float()
    

    batch_size = 5


    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(train_x,train_y)

    # data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)

    return train_loader,len(X_train),mapping_entities,mapping_properties

class AutoEncoder(nn.Module):
    def __init__(self,trial,max_e,max_p):
        super(AutoEncoder, self).__init__()
        self.down_layers=nn.ModuleList()
        self.up_layers=nn.ModuleList()
        self.down_pool_layers=nn.ModuleList()
        self.up_pool_layers=nn.ModuleList()
        self.n_layers = trial.suggest_int("n_layers", 1, 5)
        self.chosen_params=[]
        for i in range(self.n_layers):
            
          #Encoder
            if(i==0):
                #self.chosen_params.append([trial.suggest_int("n_filter_l{}".format(i), 1, max_p),trial.suggest_int("kernel_size_l{}".format(i), 2, max_e)])
                #self.down_layers.append(nn.Conv3d(1,1,self.chosen_params[i][1]))
                #self.chosen_params.append()
                kernel_size_wh = trial.suggest_int("kernel_size_l{}".format(i),3,15,2)
                self.chosen_params.append([5,1,(kernel_size_wh,kernel_size_wh,1)])
                self.down_layers.append(nn.Conv3d(5,1,(kernel_size_wh,kernel_size_wh,1)))
            else:
                kernel_size_wh = trial.suggest_int("kernel_size_l{}".format(i),3,15,2)
                self.chosen_params.append([self.chosen_params[i-1][1],trial.suggest_int("n_filter_l{}".format(i), self.chosen_params[i-1][1], self.chosen_params[i-1][1]*2),(kernel_size_wh,kernel_size_wh,1)])
                self.down_layers.append(nn.Conv3d(self.chosen_params[i][0],self.chosen_params[i][1],self.chosen_params[i][2]))
            if(i!=(self.n_layers-1)):
                self.down_pool_layers.append(nn.MaxPool3d(kernel_size=2, stride=trial.suggest_int("stride_pool_{}".format(i), 1, 2)))
        for i in range(self.n_layers):
          #Encoder
            if(i!=(self.n_layers-1)):
                self.up_layers.append(nn.ConvTranspose3d(self.chosen_params[len(self.chosen_params)-1-i][1],self.chosen_params[len(self.chosen_params)-1-i][0],self.chosen_params[len(self.chosen_params)-1-i][2]))
            else:
                #self.up_layers.append(nn.ConvTranspose3d(self.chosen_params[len(self.chosen_params)-1-i][0],1,self.chosen_params[len(self.chosen_params)-1-i][1]))
                self.up_layers.append(nn.ConvTranspose3d(1,5,self.chosen_params[len(self.chosen_params)-1-i][2]))
            if(i!=(self.n_layers-1)):            
                self.up_pool_layers.append(nn.MaxUnpool3d(kernel_size=2, stride=trial.suggest_int("stride_pool_{}".format(i), 1, 2)))
        for el in self.down_layers:
            torch.nn.init.normal_(el.weight, mean=0.5, std=0.7)
        for el in self.up_layers:
            torch.nn.init.normal_(el.weight, mean=0.5, std=0.7)

    def encode(self, x, return_partials=True):
        # Encoder
        up_shapes=[]
        idxs=[]
        for i in range(len(self.down_layers)): 
            x = self.down_layers[i](x)
            x = torch.sigmoid(x)
            #up_shapes.append(x.shape)
            #x,idx = self.down_pool_layers[i](x)
            #idxs,append(idx)

        """   
        up3out_shape = x.shape
        x, i1 = self.pool1(x)

        x = self.conv2(x)
        up2out_shape = x.shape
        x, i2 = self.pool2(x)

        x = self.conv3(x)
        up1out_shape = x.shape
        x, i3 = self.pool3(x)

        x = self.conv4(x)
        up0out_shape = x.shape
        x, i4 = self.pool4(x)

        x = x.view(-1, 10 * 10)
        x = F.relu(self.fc1(x))
        """
        if return_partials:
            return x, up_shapes, idxs

        else:
            return x

    def forward(self, x):

        # Decoder
        x, up_shapes, idxs = self.encode(x)
        for i in range(len(self.up_layers)):
            #x = self.up_pool_layers[i](x, output_size=up_shapes[len(up_shapes)-1-i], indices=idxs[len(idxs)-1-i])
            x = self.up_layers[i](x)
            x = torch.sigmoid(x)
        
        return x

def objective_max_accuracy(trial):
    max_e,max_p = 353,11
    DEVICE = get_device()
    EPOCHS = 2
    BATCHSIZE=5
    criterion = torch.nn.BCELoss()
    mapping_entities={}
    mapping_properties={}
    try: 
        # Generate the model.
        model = AutoEncoder(trial,max_e,max_p).to(DEVICE)
        #print(model)
        # Generate the optimizers.
        lr = 0.09
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

        index = random.randint(0,9)
        # Training of the model.
        for epoch in range(EPOCHS):
            model.train()
            for i in range(8):
                train_loader,X_train_shape,mapping_entities,mapping_properties= get_dataset(i,index,mapping_entities,mapping_properties)
                N_TRAIN_EXAMPLES = X_train_shape
                for batch_idx, (data, target) in enumerate(train_loader):
                    # Limiting training data for faster epochs.
                    if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                        break
                    data, target = data[None, ...].to(DEVICE, dtype=torch.float), target[None, ...].to(DEVICE, dtype=torch.float)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                del train_loader
                gc.collect()
            # Validation of the model.
            model.eval()
            correct = 0
            tot = 0
            
            with torch.no_grad():
                for i in range(8,10):
                    valid_loader,X_valid_shape,mapping_entities,mapping_properties= get_dataset(i,index,mapping_entities,mapping_properties)
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
                    del valid_loader
                    gc.collect()
            accuracy = correct*100 / tot
            trial.report(accuracy, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return accuracy
    except Exception as e:
        print(e)
        trial.report(0, epoch)
        return 0




if __name__ == "__main__":
    study_max_acc = optuna.create_study(direction="maximize")
    study_max_acc.optimize(objective_max_accuracy, n_trials=10000, timeout=None,gc_after_trial=True)

    pruned_trials = study_max_acc.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study_max_acc.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study_max_acc.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial maximizing accuracy:")
    print("Best trial:")
    trial = study_max_acc.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

