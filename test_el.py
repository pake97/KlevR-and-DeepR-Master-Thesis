import torch 
import torch.nn as nn
import numpy as np
import gc
import pandas as pd
from rdflib import Graph,Namespace
import json 
import csv
from time import time
import warnings
def getIndex(e,mapping):
    if(e.split('/')[-1].split('#')[-1] in mapping.keys()):
        return mapping.get(e.split('/')[-1].split('#')[-1])
    if(e.split('/')[-1].split('#')[-1] not in mapping.keys()):
        return -1

def embed(mapping_e, mapping_p, graph,max_e,max_p):
    tensor = np.zeros((max_e,max_e,max_p), dtype=np.int32)
    for s,p,o in graph.triples((None,None,None)):
        idx_s=getIndex(s,mapping_e)
        idx_o=getIndex(o,mapping_e)
        idx_p=getIndex(p,mapping_p)
        if(idx_s==-1 or idx_o==-1 or idx_p==-1):
            ciao=0
        else:
            tensor[idx_s][idx_o][idx_p]=1
    return tensor

class CustomEvaluation():
    def __init__(self,mapping_e,mapping_p,max_e,max_p,batch_size):
        
        self.tp_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/el/test/tp.csv"
        self.tn_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/el/test/tn.csv"
        self.fp_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/el/test/fp.csv"
        self.fn_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/el/test/fn.csv"
        self.xtp_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/el/test/xtp.csv"
        self.xtn_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/el/test/xtn.csv"
        self.xfp_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/el/test/xfp.csv"
        self.xfn_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/el/test/xfn.csv"
        self.pred_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/el/test/pred.csv"
        self.agg_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/el/test/agg.csv"
        self.tolte_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/el/test/tolte.csv"
        self.absolute_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/el/test/"
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


def get_dataset(i,mapping_entities,mapping_properties,max_e,max_p):
    max_e,max_p=max_e,max_p
    X_train=[]
    Y_train=[]
    for j in range(i*150,(i+1)*150):
        n = Namespace("http://example.org/")
        g1 = Graph()
        g1.bind("ex",n)
        g1.parse("/home/ubuntu/data/home/ubuntu/deeplogic/src/main/resources/dataset/dataset/valid/el/scene{}.ttl".format(j), format="xml")
        spo_tensor = embed(mapping_entities,mapping_properties,g1,max_e,max_p)
        X_train.append(spo_tensor)
        g2 = Graph()
        g2.bind("ex",n)
        g2.parse("/home/ubuntu/data/home/ubuntu/deeplogic/src/main/resources/dataset/dataset/valid_materialized/el/scene{}.ttl".format(j),format="turtle")

        g3 = Graph()
        g3.parse("/home/ubuntu/data/home/ubuntu/deeplogic/clevr_el_materialized.ttl", format="ttl")

        graph = g2 - g3
        graph.bind("ex",n)

        spo_tensor = embed(mapping_entities,mapping_properties,graph,max_e,max_p)
        Y_train.append(spo_tensor)
        #counter = counter +1
        #if(counter ==1000):
        #break
        
        print(j)
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
    try: 
        model = AutoEncoder(options,max_e,max_p,BATCHSIZE).to(DEVICE)
        model.load_state_dict(torch.load("/home/ubuntu/data/home/ubuntu/deeplogic/el_model_params/eporch0")['model_state_dict'])
        model.eval()
        correct = 0
        tot = 0
        ti=0
        with torch.no_grad():
            for i in range(100):
                valid_loader,X_valid_shape= get_dataset(i,mapping_entities,mapping_properties,max_e,max_p)
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
    options=[[9,1,2],[11,2,1],[9,3,2],[7,4,1]]
    mapping_entities={}
    mapping_properties={}
    # Opening JSON file
    with open('/home/ubuntu/data/home/ubuntu/deeplogic/el_model_params/mapping_e.json') as json_file:
        mapping_entities = json.load(json_file)
    with open('/home/ubuntu/data/home/ubuntu/deeplogic/el_model_params/mapping_p.json') as json_file:
        mapping_properties = json.load(json_file)
    warnings.filterwarnings("ignore")
    mapping_entities={"o4": 0, "o0": 1, "o1": 2, "o3": 3, "BrownMetallicCylinderLargeObject": 4, "GreenMetallicCylinderLargeObject": 5, "o5": 6, "o2": 7, "CyanRubberCubeSmallObject": 8, "GrayRubberCubeSmallObject": 9, "BlueRubberCubeLargeObject": 10, "BrownMetallicSphereLargeObject": 11, "ColoredMaterialShapedObject": 12, "ShapedObject": 13, "CyanRubberSmallObject": 14, "CyanRubberObject": 15, "CyanCubeSmallObject": 16, "BrownMetallicLargeObject": 17, "BrownLargeObject": 18, "RubberCubeObject": 19, "MaterialSizedObject": 20, "BrownMetallicObject": 21, "MetallicCylinderObject": 22, "RubberLargeObject": 23, "ColoredSizedObject": 24, "CylinderObject": 25, "Object": 26, "CylinderLargeObject": 27, "SphereObject": 28, "ColoredShapedSizedObject": 29, "SmallObject": 30, "BlueCubeObject": 31, "MaterialShapedSizedObject": 32, "MetallicLargeObject": 33, "SizedObject": 34, "MaterialShapedObject": 35, "ColoredMaterialObject": 36, "BrownMetallicCylinderObject": 37, "BrownCylinderObject": 38, "ColoredMaterialShapedSizedObject": 39, "BrownCylinderLargeObject": 40, "ColoredShapedObject": 41, "BlueRubberLargeObject": 42, "GreenLargeObject": 43, "CubeObject": 44, "SphereLargeObject": 45, "GrayCubeObject": 46, "NamedIndividual": 47, "MetallicCylinderLargeObject": 48, "BrownSphereObject": 49, "BrownObject": 50, "ColoredMaterialSizedObject": 51, "CubeSmallObject": 52, "LargeObject": 53, "ShapedSizedObject": 54, "MaterialObject": 55, "GreenCylinderLargeObject": 56, "RubberObject": 57, "ColoredObject": 58, "Thing": 59, "CubeLargeObject": 60, "RubberSmallObject": 61, "BrownMetallicSphereObject": 62, "BlueCubeLargeObject": 63, "CyanSmallObject": 64, "BrownSphereLargeObject": 65, "GreenObject": 66, "MetallicSphereObject": 67, "MetallicObject": 68, "GrayObject": 69, "CyanObject": 70, "GreenCylinderObject": 71, "MetallicSphereLargeObject": 72, "CyanRubberCubeObject": 73, "RubberCubeSmallObject": 74, "GraySmallObject": 75, "BlueLargeObject": 76, "BlueRubberObject": 77, "GrayRubberObject": 78, "GrayRubberCubeObject": 79, "GrayRubberSmallObject": 80, "GreenMetallicLargeObject": 81, "BlueObject": 82, "GrayCubeSmallObject": 83, "GreenMetallicCylinderObject": 84, "GreenMetallicObject": 85, "BlueRubberCubeObject": 86, "CyanCubeObject": 87, "RubberCubeLargeObject": 88, "o8": 89, "o6": 90, "YellowMetallicSphereSmallObject": 91, "YellowRubberCubeSmallObject": 92, "o7": 93, "GreenRubberSphereSmallObject": 94, "BrownRubberCubeSmallObject": 95, "GrayRubberCubeLargeObject": 96, "YellowRubberCylinderSmallObject": 97, "BlueRubberSphereLargeObject": 98, "RedRubberSphereSmallObject": 99, "GrayMetallicSphereSmallObject": 100, "RedSphereObject": 101, "YellowRubberObject": 102, "GreenRubberSmallObject": 103, "SphereSmallObject": 104, "MetallicSmallObject": 105, "BrownRubberCubeObject": 106, "GreenSphereSmallObject": 107, "YellowSphereSmallObject": 108, "RubberCylinderObject": 109, "MetallicSphereSmallObject": 110, "RedSphereSmallObject": 111, "GreenSphereObject": 112, "GreenRubberSphereObject": 113, "GrayMetallicObject": 114, "BlueSphereLargeObject": 115, "RubberSphereObject": 116, "BlueSphereObject": 117, "YellowMetallicObject": 118, "YellowRubberCylinderObject": 119, "GreenRubberObject": 120, "GrayCubeLargeObject": 121, "BlueRubberSphereObject": 122, "RedSmallObject": 123, "RedRubberSphereObject": 124, "YellowRubberSmallObject": 125, "YellowSmallObject": 126, "YellowRubberCubeObject": 127, "YellowCylinderSmallObject": 128, "RedRubberObject": 129, "GrayMetallicSphereObject": 130, "GrayLargeObject": 131, "BrownCubeObject": 132, "YellowObject": 133, "RubberSphereSmallObject": 134, "YellowSphereObject": 135, "GrayMetallicSmallObject": 136, "BrownSmallObject": 137, "YellowCubeObject": 138, "GraySphereObject": 139, "RedObject": 140, "GreenSmallObject": 141, "YellowMetallicSphereObject": 142, "RubberSphereLargeObject": 143, "GrayRubberLargeObject": 144, "RubberCylinderSmallObject": 145, "BrownRubberSmallObject": 146, "YellowCubeSmallObject": 147, "YellowCylinderObject": 148, "BrownRubberObject": 149, "CylinderSmallObject": 150, "GraySphereSmallObject": 151, "BrownCubeSmallObject": 152, "RedRubberSmallObject": 153, "YellowMetallicSmallObject": 154, "YellowMetallicSphereLargeObject": 155, "YellowSphereLargeObject": 156, "YellowLargeObject": 157, "YellowMetallicLargeObject": 158, "CyanMetallicSphereLargeObject": 159, "o9": 160, "PurpleRubberCubeSmallObject": 161, "CyanMetallicCylinderLargeObject": 162, "BlueMetallicCubeSmallObject": 163, "GrayMetallicCylinderLargeObject": 164, "GreenRubberSphereLargeObject": 165, "PurpleRubberObject": 166, "PurpleCubeObject": 167, "CyanMetallicObject": 168, "BlueMetallicObject": 169, "GrayCylinderObject": 170, "BlueCubeSmallObject": 171, "MetallicCubeSmallObject": 172, "CyanCylinderObject": 173, "PurpleObject": 174, "CyanSphereLargeObject": 175, "GrayMetallicLargeObject": 176, "CyanMetallicCylinderObject": 177, "PurpleSmallObject": 178, "PurpleCubeSmallObject": 179, "PurpleRubberSmallObject": 180, "PurpleRubberCubeObject": 181, "CyanMetallicLargeObject": 182, "GreenRubberLargeObject": 183, "BlueMetallicCubeObject": 184, "CyanCylinderLargeObject": 185, "GrayCylinderLargeObject": 186, "CyanSphereObject": 187, "GrayMetallicCylinderObject": 188, "CyanLargeObject": 189, "GreenSphereLargeObject": 190, "MetallicCubeObject": 191, "BlueMetallicSmallObject": 192, "CyanMetallicSphereObject": 193, "BlueSmallObject": 194, "PurpleMetallicSphereSmallObject": 195, "CyanMetallicCylinderSmallObject": 196, "PurpleSphereObject": 197, "PurpleMetallicSphereObject": 198, "PurpleMetallicSmallObject": 199, "PurpleMetallicObject": 200, "MetallicCylinderSmallObject": 201, "CyanCylinderSmallObject": 202, "CyanMetallicSmallObject": 203, "PurpleSphereSmallObject": 204, "GreenMetallicCubeLargeObject": 205, "CyanRubberSphereSmallObject": 206, "GrayMetallicSphereLargeObject": 207, "YellowRubberCubeLargeObject": 208, "BlueRubberCylinderLargeObject": 209, "RedMetallicCubeSmallObject": 210, "YellowRubberSphereSmallObject": 211, "PurpleRubberCubeLargeObject": 212, "BrownRubberCylinderLargeObject": 213, "BlueCylinderObject": 214, "YellowCubeLargeObject": 215, "GreenMetallicCubeObject": 216, "RedMetallicSmallObject": 217, "CyanSphereSmallObject": 218, "BrownRubberCylinderObject": 219, "MetallicCubeLargeObject": 220, "BrownRubberLargeObject": 221, "PurpleLargeObject": 222, "CyanRubberSphereObject": 223, "GreenCubeObject": 224, "PurpleCubeLargeObject": 225, "RedMetallicObject": 226, "RedCubeSmallObject": 227, "YellowRubberLargeObject": 228, "RedCubeObject": 229, "RubberCylinderLargeObject": 230, "PurpleRubberLargeObject": 231, "YellowRubberSphereObject": 232, "RedMetallicCubeObject": 233, "GreenCubeLargeObject": 234, "GraySphereLargeObject": 235, "BlueCylinderLargeObject": 236, "BlueRubberCylinderObject": 237, "GreenRubberCubeSmallObject": 238, "GreenRubberCubeObject": 239, "GreenCubeSmallObject": 240, "PurpleMetallicCubeLargeObject": 241, "GreenMetallicCubeSmallObject": 242, "BlueRubberSphereSmallObject": 243, "BlueRubberCubeSmallObject": 244, "BlueRubberSmallObject": 245, "PurpleMetallicLargeObject": 246, "PurpleMetallicCubeObject": 247, "GreenMetallicSmallObject": 248, "BlueSphereSmallObject": 249, "BrownRubberSphereSmallObject": 250, "BrownSphereSmallObject": 251, "BrownRubberSphereObject": 252, "CyanRubberCylinderSmallObject": 253, "CyanRubberSphereLargeObject": 254, "YellowMetallicCylinderSmallObject": 255, "GrayRubberCylinderSmallObject": 256, "GrayRubberCylinderObject": 257, "YellowMetallicCylinderObject": 258, "CyanRubberCylinderObject": 259, "GrayCylinderSmallObject": 260, "CyanRubberLargeObject": 261, "RedMetallicSphereLargeObject": 262, "GreenMetallicCylinderSmallObject": 263, "GreenRubberCylinderSmallObject": 264, "RedMetallicCylinderSmallObject": 265, "RedMetallicCubeLargeObject": 266, "RedMetallicSphereObject": 267, "RedLargeObject": 268, "RedMetallicLargeObject": 269, "RedCylinderSmallObject": 270, "GreenCylinderSmallObject": 271, "RedMetallicCylinderObject": 272, "RedCylinderObject": 273, "RedSphereLargeObject": 274, "RedCubeLargeObject": 275, "GreenRubberCylinderObject": 276, "PurpleMetallicCylinderSmallObject": 277, "PurpleMetallicSphereLargeObject": 278, "GreenMetallicSphereLargeObject": 279, "CyanMetallicCubeSmallObject": 280, "PurpleRubberCylinderLargeObject": 281, "BlueMetallicCylinderLargeObject": 282, "PurpleMetallicCylinderObject": 283, "PurpleCylinderObject": 284, "PurpleRubberCylinderObject": 285, "GreenMetallicSphereObject": 286, "BlueMetallicCylinderObject": 287, "BlueMetallicLargeObject": 288, "CyanMetallicCubeObject": 289, "PurpleCylinderSmallObject": 290, "PurpleCylinderLargeObject": 291, "PurpleSphereLargeObject": 292, "BlueMetallicCylinderSmallObject": 293, "BlueCylinderSmallObject": 294, "RedMetallicCylinderLargeObject": 295, "YellowMetallicCylinderLargeObject": 296, "RedCylinderLargeObject": 297, "YellowCylinderLargeObject": 298, "RedRubberCubeSmallObject": 299, "GreenMetallicSphereSmallObject": 300, "RedRubberCubeObject": 301, "BrownRubberSphereLargeObject": 302, "GrayMetallicCylinderSmallObject": 303, "GrayMetallicCubeLargeObject": 304, "BrownMetallicCubeLargeObject": 305, "BrownMetallicSphereSmallObject": 306, "YellowRubberCylinderLargeObject": 307, "BrownMetallicCubeObject": 308, "BrownCubeLargeObject": 309, "GrayMetallicCubeObject": 310, "BrownMetallicSmallObject": 311, "BlueMetallicSphereLargeObject": 312, "PurpleMetallicCylinderLargeObject": 313, "BlueMetallicSphereObject": 314, "RedMetallicSphereSmallObject": 315, "YellowMetallicCubeSmallObject": 316, "GreenRubberCubeLargeObject": 317, "CyanMetallicCubeLargeObject": 318, "CyanCubeLargeObject": 319, "YellowMetallicCubeObject": 320, "PurpleMetallicCubeSmallObject": 321, "GrayRubberSphereSmallObject": 322, "GrayRubberSphereObject": 323, "BlueMetallicSphereSmallObject": 324, "GrayRubberCylinderLargeObject": 325, "PurpleRubberCylinderSmallObject": 326, "RedRubberSphereLargeObject": 327, "RedRubberLargeObject": 328, "BlueRubberCylinderSmallObject": 329, "RedRubberCubeLargeObject": 330, "CyanMetallicSphereSmallObject": 331, "PurpleRubberSphereSmallObject": 332, "PurpleRubberSphereObject": 333, "RedRubberCylinderSmallObject": 334, "YellowRubberSphereLargeObject": 335, "RedRubberCylinderObject": 336, "GrayMetallicCubeSmallObject": 337, "YellowMetallicCubeLargeObject": 338, "GrayRubberSphereLargeObject": 339, "GreenRubberCylinderLargeObject": 340, "BrownRubberCylinderSmallObject": 341, "BrownCylinderSmallObject": 342, "BrownMetallicCubeSmallObject": 343, "BlueMetallicCubeLargeObject": 344, "CyanRubberCylinderLargeObject": 345, "CyanRubberCubeLargeObject": 346, "PurpleRubberSphereLargeObject": 347, "BrownMetallicCylinderSmallObject": 348, "BrownRubberCubeLargeObject": 349, "RedRubberCylinderLargeObject": 350}
    mapping_properties={"hasDirectlyBehind": 0, "hasDirectlyOnFront": 1, "hasDirectlyOnRight": 2, "type": 3, "hasDirectlyOnLeft": 4, "hasNear": 5, "hasDirectlyNear": 6, "hasOnRight": 7, "hasOnFront": 8, "hasOnLeft": 9, "hasBehind": 10}
    test_model(options,351,11,mapping_entities,mapping_properties)
    #ql
    """
     {'n_layers': 1, 'kernel_size_l0': 9}
    """
    #options=[[9,1,1]]
    #train_model(options,353,11)




