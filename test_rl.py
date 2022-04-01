import torch 
import torch.nn as nn
import numpy as np
import gc
import pandas as pd
from rdflib import Graph,Namespace
import json 
import csv
from time import time

def getIndex(e,mapping):
    if(e.split('/')[-1].split('#')[-1] in mapping.keys()):
        return mapping.get(e.split('/')[-1].split('#')[-1])
    if(e.split('/')[-1].split('#')[-1] not in mapping.keys()):
        mapping[e.split('/')[-1].split('#')[-1]]=len(mapping.keys())
        return mapping.get(e.split('/')[-1].split('#')[-1])

def embed(mapping_e, mapping_p, graph,max_e,max_p):
    tensor = np.zeros((max_e,max_e,max_p), dtype=np.int32)
    n = Namespace("http://example.org/")
    q_n_blank = graph.query('SELECT ?s ?p ?o WHERE {?s ?p ?o filter(!isBlank(?s) && !isBlank(?o))}',initNs={ 'ex': n })
    #for s,p,o in graph.triples((None,None,None)):
    for row in q_n_blank:
        idx_s=getIndex(row.s,mapping_e)
        idx_o=getIndex(row.o,mapping_e)
        idx_p=getIndex(row.p,mapping_p)
        tensor[idx_s][idx_o][idx_p]=1
    return tensor

class CustomEvaluation():
    def __init__(self,mapping_e,mapping_p,max_e,max_p,batch_size):
        
        self.tp_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/rl/test/tp.csv"
        self.tn_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/rl/test/tn.csv"
        self.fp_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/rl/test/fp.csv"
        self.fn_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/rl/test/fn.csv"
        self.xtp_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/rl/test/xtp.csv"
        self.xtn_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/rl/test/xtn.csv"
        self.xfp_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/rl/test/xfp.csv"
        self.xfn_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/rl/test/xfn.csv"
        self.pred_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/rl/test/pred.csv"
        self.agg_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/rl/test/agg.csv"
        self.tolte_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/rl/test/tolte.csv"
        self.absolute_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/rl/test/"
        self.tot = max_e*max_e*max_p*batch_size
        self.max_e=max_e
        self.max_p=max_p
        self.batch_size=batch_size
        self.mapping_p=mapping_p
        self.mapping_e=mapping_e
            
        

    def update(self,ny_pred_batch,ny_true_batch,x):
        y_pred_batch = ny_pred_batch - x
        y_true_batch = ny_true_batch - x

        for i in range(self.batch_size):
            #accuracy=corrects/self.tot
            #self.write_csv(self.accuracy_path,accuracy)
            y_true = y_true_batch[0][i]
            y_pred = y_pred_batch[0][i]
            z_pred = ny_pred_batch[0][i]
            z_true = ny_true_batch[0][i]
            xxi=x[0][i]
            tn,fp,fn,tp = 0,0,0,0
            ztp,zfp,zfn,ztn=0,0,0,0
            aggiunte = 0
            pred_aggiunte=0
            pred_tolte=0
            for row in range(self.max_e):
                for col in range(self.max_e):
                    for sli in range(self.max_p):
                        #print(y_pred[row][col][sli].item(),flush=True)
                        #print(y_true[row][col][sli].item(),flush=True)
                        yp=y_pred[row][col][sli].item()
                        yt=y_true[row][col][sli].item()
                        zp=z_pred[row][col][sli].item()
                        zt=z_true[row][col][sli].item()
                        xi = xxi[row][col][sli].item()

                        if((yp==0) and (yt==0)):
                            tn = tn+1
                        if((yp==0) and (yt==1)):
                            fn = fn+1
                        if((yp==1) and (yt==1)):
                            tp = tp+1
                        if((yp==1) and (yt==0)):
                            fp = fp+1
                        if((zp==0) and (zt==0)):
                            ztn = ztn+1
                        if((zp==0) and (zt==1)):
                            zfn = zfn+1
                            if(xi==1):
                                pred_tolte = pred_tolte+1
                            else:
                                aggiunte = aggiunte+1
                        if((zp==1) and (zt==1)):
                            ztp = ztp+1
                            if(xi==1):
                                aggiunte = aggiunte+1
                                pred_aggiunte=pred_aggiunte+1
                        if((zp==1) and (zt==0)):
                            zfp = zfp+1

            self.write_csv(self.tp_path,tp)
            self.write_csv(self.fp_path,fp)
            self.write_csv(self.tn_path,tn)
            self.write_csv(self.fn_path,fn)
            self.write_csv(self.xtp_path,ztp)
            self.write_csv(self.xfp_path,zfp)
            self.write_csv(self.xtn_path,ztn)
            self.write_csv(self.xfn_path,zfn)
            self.write_csv(self.agg_path,aggiunte)
            self.write_csv(self.pred_path,pred_aggiunte)
            self.write_csv(self.tolte_path,pred_tolte)
            

       
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


def get_dataset(i,mapping_entities,mapping_properties,max_e,max_p,file):
    max_e,max_p=max_e,max_p
    X_train=[]
    Y_train=[]
    for k in range(i*150,(i+1)*150):
        j=file[k]
        n = Namespace("http://example.org/")
        g1 = Graph()
        g1.bind("ex",n)
        g1.parse("/home/ubuntu/data/home/ubuntu/deeplogic/train_pre/scene{}.ttl".format(j), format="turtle")
        spo_tensor = embed(mapping_entities,mapping_properties,g1,max_e,max_p)
        #counter = counter +1
        X_train.append(spo_tensor)
        #if(counter ==1000):
        #break
        #counter = 0
        #for filename in os.listdir("/home/ubuntu/data/home/ubuntu/deeplogic/el_dataset/y"):
        g2 = Graph()
        g2.bind("ex",n)
        g2.parse("/home/ubuntu/data/home/ubuntu/deeplogic/src/main/resources/dataset/dataset/train_materialized/rl/scene{}.ttl".format(j),format="turtle")

        g3 = Graph()
        g3.parse("/home/ubuntu/data/home/ubuntu/deeplogic/clevr_rl_materialized.ttl", format="ttl")

        graph = g2 - g3
        graph.bind("ex",n)

        spo_tensor = embed(mapping_entities,mapping_properties,graph,max_e,max_p)
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

    return train_loader,len(X_train)

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

def test_model(options,num_e,num_p,mapping_entities,mapping_properties):
    max_e,max_p = num_e,num_p
    DEVICE = get_device()
    BATCHSIZE=5
    evaluation = CustomEvaluation(mapping_entities,mapping_properties,max_e,max_p,BATCHSIZE)
    mypath="/home/ubuntu/data/home/ubuntu/deeplogic/src/main/resources/dataset/dataset/train_materialized/rl/"
    files=[]
    with open('/home/ubuntu/data/home/ubuntu/deeplogic//used_rl.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            files.append(row)

    final = [f.strip('\'').strip(' \'')  for f in files[0]] 
    final=final[5000:6500]
    try: 
        model = AutoEncoder(options,max_e,max_p,BATCHSIZE).to(DEVICE)
        model.load_state_dict(torch.load("/home/ubuntu/data/home/ubuntu/deeplogic/rl_model_params/epochs1")['model_state_dict'])
        model.eval()
        correct = 0
        tot = 0
        ti=0
        with torch.no_grad():
            for i in range(10):
                print(i)
                valid_loader,X_valid_shape= get_dataset(i,mapping_entities,mapping_properties,max_e,max_p,final)
                N_VALID_EXAMPLES = X_valid_shape
                t0=time()
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
                    evaluation.update(pred,target,data)
                t1=time()
                del valid_loader
                gc.collect()
        accuracy = correct*100 / tot
        print("ACCURACY: " +str(accuracy),flush=True)
        print("time: " + str(ti))
    except Exception as e:
        print(e)



if __name__ == "__main__":
    #el
    """
    {'n_layers': 4, 'kernel_size_l0': 9, 'stride_pool_0': 2, 'kernel_size_l1': 11, 'n_filter_l1': 2, 'stride_pool_1': 1, 'kernel_size_l2': 9, 'n_filter_l2': 3, 'stride_pool_2': 2, 'kernel_size_l3': 7, 'n_filter_l3': 4}
    """
    
    mapping_entities = {"Sphere": 0, "o7": 1, "o2": 2, "Gray": 3, "o4": 4, "o0": 5, "o5": 6, "ShapedObject": 7, "Yellow": 8, "Brown": 9, "Metal": 10, "CylinderLargeObject": 11, "o3": 12, "Small": 13, "Cylinder": 14, "ShapedSizedObject": 15, "Purple": 16, "o6": 17, "Red": 18, "o1": 19, "NamedIndividual": 20, "Big": 21, "BlueCubeLargeObject": 22, "Cube": 23, "ColoredMaterialShapedObject": 24, "ColoredSizedObject": 25, "MaterialObject": 26, "RubberObject": 27, "GrayCubeObject": 28, "SphereObject": 29, "YellowLargeObject": 30, "RubberSphereObject": 31, "Cyan": 32, "SizedObject": 33, "Object": 34, "BrownMetallicSmallObject": 35, "Green": 36, "ColoredShapedSizedObject": 37, "RubberCylinderObject": 38, "ColoredObject": 39, "MaterialSizedObject": 40, "Blue": 41, "MaterialShapedObject": 42, "BrownObject": 43, "SphereSmallObject": 44, "GrayMetallicObject": 45, "CyanRubberObject": 46, "RubberLargeObject": 47, "CyanRubberSmallObject": 48, "ColoredMaterialSizedObject": 49, "ColoredMaterialShapedSizedObject": 50, "MetallicSmallObject": 51, "BrownCubeSmallObject": 52, "LargeObject": 53, "test": 54, "RedObject": 55, "BlueRubberCubeObject": 56, "GrayLargeObject": 57, "CubeObject": 58, "SmallObject": 59, "RubberCubeLargeObject": 60, "Thing": 61, "GrayRubberCylinderObject": 62, "MaterialShapedSizedObject": 63, "CubeLargeObject": 64, "BlueObject": 65, "RubberCubeObject": 66, "CubeSmallObject": 67, "GrayCubeSmallObject": 68, "Rubber": 69, "RubberSmallObject": 70, "GrayRubberCylinderLargeObject": 71, "BlueCubeObject": 72, "CylinderObject": 73, "GrayObject": 74, "CyanObject": 75, "ColoredMaterialObject": 76, "GrayRubberSmallObject": 77, "GrayCylinderObject": 78, "YellowRubberLargeObject": 79, "GrayRubberSphereSmallObject": 80, "RedRubberObject": 81, "GrayMetallicCubeSmallObject": 82, "BlueRubberObject": 83, "GrayMetallicCubeObject": 84, "CyanRubberSphereObject": 85, "ColoredShapedObject": 86, "GraySmallObject": 87, "BlueRubberLargeObject": 88, "GraySphereObject": 89, "RedSphereSmallObject": 90, "CyanRubberSphereSmallObject": 91, "RubberSphereSmallObject": 92, "GrayRubberSphereObject": 93, "BlueRubberCubeLargeObject": 94, "RedSmallObject": 95, "RedRubberSphereSmallObject": 96, "RedSphereObject": 97, "CyanSmallObject": 98, "RedRubberSphereObject": 99, "MetallicCubeSmallObject": 100, "CyanSphereSmallObject": 101, "GrayRubberObject": 102, "MetallicCubeObject": 103, "RedRubberSmallObject": 104, "YellowRubberCubeLargeObject": 105, "BrownMetallicObject": 106, "BrownMetallicCubeSmallObject": 107, "BlueLargeObject": 108, "BrownCubeObject": 109, "YellowRubberObject": 110, "CyanSphereObject": 111, "GrayCylinderLargeObject": 112, "GrayMetallicSmallObject": 113, "RubberCylinderLargeObject": 114, "GraySphereSmallObject": 115, "YellowRubberCubeObject": 116, "BrownMetallicCubeObject": 117, "MetallicObject": 118, "BrownSmallObject": 119, "YellowCubeLargeObject": 120, "YellowObject": 121, "YellowCubeObject": 122, "GrayRubberLargeObject": 123, "YellowMetallicSmallObject": 124, "YellowSphereObject": 125, "MetallicSphereSmallObject": 126, "YellowMetallicSphereObject": 127, "YellowMetallicObject": 128, "YellowSmallObject": 129, "YellowMetallicSphereSmallObject": 130, "YellowSphereSmallObject": 131, "MetallicSphereObject": 132, "o8": 133, "o9": 134, "GreenRubberCubeObject": 135, "GreenRubberCubeSmallObject": 136, "PurpleMetallicSphereSmallObject": 137, "PurpleMetallicObject": 138, "YellowCylinderLargeObject": 139, "GreenSmallObject": 140, "PurpleMetallicSphereObject": 141, "GrayMetallicCylinderSmallObject": 142, "CylinderSmallObject": 143, "RubberCubeSmallObject": 144, "YellowCylinderSmallObject": 145, "GreenRubberSmallObject": 146, "YellowRubberCylinderSmallObject": 147, "MetallicCylinderSmallObject": 148, "PurpleMetallicSmallObject": 149, "GreenObject": 150, "YellowMetallicLargeObject": 151, "PurpleMetallicCubeSmallObject": 152, "PurpleObject": 153, "PurpleMetallicCylinderLargeObject": 154, "GrayCylinderSmallObject": 155, "MetallicCylinderObject": 156, "YellowRubberCylinderObject": 157, "YellowRubberCubeSmallObject": 158, "YellowRubberSmallObject": 159, "PurpleSmallObject": 160, "YellowMetallicCylinderObject": 161, "PurpleCylinderLargeObject": 162, "PurpleCubeSmallObject": 163, "GreenMetallicSmallObject": 164, "GreenCylinderSmallObject": 165, "PurpleSphereSmallObject": 166, "MetallicCylinderLargeObject": 167, "PurpleMetallicCubeObject": 168, "YellowCubeSmallObject": 169, "GreenCubeSmallObject": 170, "GrayMetallicCylinderObject": 171, "YellowCylinderObject": 172, "GreenCylinderObject": 173, "PurpleMetallicLargeObject": 174, "GreenMetallicCylinderSmallObject": 175, "PurpleLargeObject": 176, "GreenRubberObject": 177, "RubberCylinderSmallObject": 178, "GreenCubeObject": 179, "MetallicLargeObject": 180, "YellowMetallicCylinderLargeObject": 181, "GreenMetallicObject": 182, "PurpleMetallicCylinderObject": 183, "PurpleCubeObject": 184, "PurpleSphereObject": 185, "PurpleCylinderObject": 186, "GreenMetallicCylinderObject": 187, "BlueSphereSmallObject": 188, "GreenMetallicCubeSmallObject": 189, "BlueRubberSphereObject": 190, "PurpleRubberObject": 191, "PurpleRubberCubeLargeObject": 192, "PurpleCubeLargeObject": 193, "PurpleRubberLargeObject": 194, "BlueSphereObject": 195, "GreenMetallicCubeObject": 196, "BlueRubberSphereSmallObject": 197, "RubberSphereLargeObject": 198, "SphereLargeObject": 199, "BlueRubberSmallObject": 200, "CyanSphereLargeObject": 201, "BlueRubberSphereLargeObject": 202, "BlueSmallObject": 203, "CyanRubberLargeObject": 204, "PurpleRubberCubeObject": 205, "CyanLargeObject": 206, "CyanRubberSphereLargeObject": 207, "BlueSphereLargeObject": 208, "BrownMetallicSphereSmallObject": 209, "BlueRubberCubeSmallObject": 210, "BrownSphereObject": 211, "GreenRubberSphereObject": 212, "GreenSphereObject": 213, "BrownSphereSmallObject": 214, "GreenSphereSmallObject": 215, "GreenRubberSphereSmallObject": 216, "BrownMetallicSphereObject": 217, "BlueCubeSmallObject": 218, "PurpleCylinderSmallObject": 219, "PurpleRubberCylinderObject": 220, "PurpleRubberSmallObject": 221, "PurpleRubberCylinderSmallObject": 222, "GreenMetallicLargeObject": 223, "BrownRubberObject": 224, "GreenLargeObject": 225, "CyanMetallicCubeSmallObject": 226, "CyanCubeObject": 227, "GreenSphereLargeObject": 228, "BrownRubberCubeObject": 229, "GreenMetallicSphereLargeObject": 230, "CyanMetallicObject": 231, "CyanMetallicCubeObject": 232, "BrownRubberSmallObject": 233, "CyanCubeSmallObject": 234, "CyanMetallicSmallObject": 235, "MetallicSphereLargeObject": 236, "BrownRubberCubeSmallObject": 237, "GreenMetallicSphereObject": 238, "GreenRubberCylinderLargeObject": 239, "GreenRubberCylinderObject": 240, "GreenRubberLargeObject": 241, "BrownRubberCylinderObject": 242, "BrownRubberCylinderLargeObject": 243, "BrownCylinderObject": 244, "PurpleRubberCubeSmallObject": 245, "BrownRubberLargeObject": 246, "BrownLargeObject": 247, "BrownCylinderLargeObject": 248, "GreenCylinderLargeObject": 249, "RedMetallicSmallObject": 250, "RedMetallicCubeSmallObject": 251, "RedMetallicCubeObject": 252, "RedCubeSmallObject": 253, "RedMetallicObject": 254, "RedCubeObject": 255, "PurpleMetallicCylinderSmallObject": 256, "GrayMetallicSphereObject": 257, "GreenRubberSphereLargeObject": 258, "GrayMetallicSphereSmallObject": 259, "CyanRubberCylinderObject": 260, "CyanRubberCylinderSmallObject": 261, "CyanCylinderSmallObject": 262, "CyanCylinderObject": 263, "RedCylinderLargeObject": 264, "RedLargeObject": 265, "RedRubberCylinderObject": 266, "RedRubberSphereLargeObject": 267, "GrayMetallicCylinderLargeObject": 268, "RedCylinderObject": 269, "RedMetallicCubeLargeObject": 270, "RedRubberLargeObject": 271, "RedSphereLargeObject": 272, "RedRubberCylinderLargeObject": 273, "RedMetallicLargeObject": 274, "MetallicCubeLargeObject": 275, "GrayMetallicLargeObject": 276, "RedCubeLargeObject": 277, "YellowMetallicSphereLargeObject": 278, "YellowMetallicCubeLargeObject": 279, "YellowMetallicCubeObject": 280, "YellowSphereLargeObject": 281, "BrownMetallicLargeObject": 282, "RedMetallicCylinderSmallObject": 283, "BrownMetallicCubeLargeObject": 284, "BrownCubeLargeObject": 285, "CyanMetallicCylinderLargeObject": 286, "RedCylinderSmallObject": 287, "CyanMetallicLargeObject": 288, "RedMetallicCylinderObject": 289, "CyanMetallicCylinderObject": 290, "CyanCylinderLargeObject": 291, "RedRubberCubeSmallObject": 292, "BrownRubberSphereObject": 293, "YellowMetallicCylinderSmallObject": 294, "BrownSphereLargeObject": 295, "RedRubberCubeObject": 296, "CyanCubeLargeObject": 297, "BrownRubberSphereLargeObject": 298, "CyanMetallicCubeLargeObject": 299, "GrayRubberSphereLargeObject": 300, "GraySphereLargeObject": 301, "BrownRubberSphereSmallObject": 302, "BlueMetallicLargeObject": 303, "PurpleRubberCylinderLargeObject": 304, "GreenCubeLargeObject": 305, "GreenMetallicCubeLargeObject": 306, "BlueMetallicObject": 307, "BlueMetallicCubeObject": 308, "BlueMetallicCubeLargeObject": 309, "CyanMetallicCylinderSmallObject": 310, "BlueMetallicCylinderSmallObject": 311, "BlueCylinderSmallObject": 312, "BlueMetallicCylinderObject": 313, "BlueCylinderObject": 314, "BlueMetallicSmallObject": 315, "GrayCubeLargeObject": 316, "BlueRubberCylinderObject": 317, "CyanRubberCubeObject": 318, "CyanRubberCubeSmallObject": 319, "GrayRubberCubeLargeObject": 320, "GrayRubberCubeObject": 321, "BlueRubberCylinderLargeObject": 322, "BlueCylinderLargeObject": 323, "YellowRubberCylinderLargeObject": 324, "RedMetallicSphereObject": 325, "YellowRubberSphereSmallObject": 326, "RedMetallicSphereLargeObject": 327, "YellowRubberSphereObject": 328, "PurpleRubberSphereObject": 329, "PurpleRubberSphereSmallObject": 330, "RedMetallicCylinderLargeObject": 331, "GrayRubberCubeSmallObject": 332, "GreenRubberCylinderSmallObject": 333, "GrayMetallicSphereLargeObject": 334, "CyanMetallicSphereObject": 335, "CyanMetallicSphereLargeObject": 336, "YellowRubberSphereLargeObject": 337, "BrownMetallicCylinderObject": 338, "BrownMetallicCylinderSmallObject": 339, "BlueMetallicCubeSmallObject": 340, "CyanRubberCylinderLargeObject": 341, "RedRubberCylinderSmallObject": 342, "BrownCylinderSmallObject": 343, "BlueMetallicCylinderLargeObject": 344, "BrownRubberCubeLargeObject": 345, "BrownMetallicSphereLargeObject": 346, "RedRubberCubeLargeObject": 347, "PurpleRubberSphereLargeObject": 348, "BrownMetallicCylinderLargeObject": 349, "PurpleSphereLargeObject": 350, "PurpleMetallicCubeLargeObject": 351, "GreenRubberCubeLargeObject": 352, "GreenMetallicCylinderLargeObject": 353, "BrownRubberCylinderSmallObject": 354, "GreenMetallicSphereSmallObject": 355, "BlueMetallicSphereObject": 356, "YellowMetallicCubeSmallObject": 357, "BlueMetallicSphereLargeObject": 358, "GrayRubberCylinderSmallObject": 359, "RedMetallicSphereSmallObject": 360, "CyanMetallicSphereSmallObject": 361, "BlueRubberCylinderSmallObject": 362, "PurpleMetallicSphereLargeObject": 363, "CyanRubberCubeLargeObject": 364, "GrayMetallicCubeLargeObject": 365, "BlueMetallicSphereSmallObject": 366,"Color":367,"Shape":368, "Size":369, "Material":370,"bBlue":371,"bBrown":372,"bCyan":373,"bGray":374,"bGreen":375,"bPurple":376,"bRed":377,"bYellow":378,"bBig":379,"bSmall":380,"bMetal":381,"bRubber":382,"bCube":383,"bSphere":384,"bCylinder":385}
    mapping_properties={"topObjectProperty": 0, "hasBehind": 1, "type": 2, "hasNear": 3, "hasDirectlyOnFront": 4, "hasOnLeft": 5, "hasDirectlyOnLeft": 6, "hasOnRight": 7, "hasOnFront": 8, "hasDirectlyOnRight": 9, "hasDirectlyNear": 10, "hasDirectlyBehind": 11,"hasColor":12,"hasShape":13,"hasMaterial":14,"hasSize":15}
   
    options=[[5,1,1],[9,2,1],[15,3,1],[13,3,2],[3,6,1]]
    test_model(options,386,16,mapping_entities,mapping_properties)
    #ql
    """
     {'n_layers': 1, 'kernel_size_l0': 9}
    """
    #options=[[9,1,1]]
    #train_model(options,353,11)




