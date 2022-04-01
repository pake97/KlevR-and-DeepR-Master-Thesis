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
        return -1

def embed(mapping_e, mapping_p, graph,max_e,max_p):
    tensor = np.zeros((max_e,max_e,max_p), dtype=np.int32)
    n = Namespace("http://example.org/")
    q_n_blank = graph.query('SELECT ?s ?p ?o WHERE {?s ?p ?o filter(!isBlank(?s) && !isBlank(?o))}',initNs={ 'ex': n })
    #for s,p,o in graph.triples((None,None,None)):
    for row in q_n_blank:
        idx_s=getIndex(row.s,mapping_e)
        idx_o=getIndex(row.o,mapping_e)
        idx_p=getIndex(row.p,mapping_p)
        if(idx_s==-1 or idx_o==-1 or idx_p==-1):
            ciao=0
        else:
            tensor[idx_s][idx_o][idx_p]=1
    return tensor

class CustomEvaluation():
    def __init__(self,mapping_e,mapping_p,max_e,max_p,batch_size):
        
        self.tp_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/ql/test/tp.csv"
        self.tn_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/ql/test/tn.csv"
        self.fp_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/ql/test/fp.csv"
        self.fn_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/ql/test/fn.csv"
        self.xtp_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/ql/test/xtp.csv"
        self.xtn_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/ql/test/xtn.csv"
        self.xfp_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/ql/test/xfp.csv"
        self.xfn_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/ql/test/xfn.csv"
        self.pred_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/ql/test/pred.csv"
        self.agg_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/ql/test/agg.csv"
        self.tolte_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/ql/test/tolte.csv"
        self.absolute_path="/home/ubuntu/data/home/ubuntu/deeplogic/finalmetrics/ql/test/"
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
        g1.parse("/home/ubuntu/data/home/ubuntu/deeplogic/src/main/resources/dataset/dataset/valid/ql/scene{}.ttl".format(j), format="xml")
        spo_tensor = embed(mapping_entities,mapping_properties,g1,max_e,max_p)
        #counter = counter +1
        X_train.append(spo_tensor)
        #if(counter ==1000):
        #break
        #counter = 0
        #for filename in os.listdir("/home/ubuntu/data/home/ubuntu/deeplogic/el_dataset/y"):
        g2 = Graph()
        g2.bind("ex",n)
        g2.parse("/home/ubuntu/data/home/ubuntu/deeplogic/src/main/resources/dataset/dataset/valid_materialized/ql/scene{}.ttl".format(j),format="turtle")

        g3 = Graph()
        g3.parse("/home/ubuntu/data/home/ubuntu/deeplogic/clevr_ql_materialized.ttl", format="ttl")

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
    try: 
        model = AutoEncoder(options,max_e,max_p,BATCHSIZE).to(DEVICE)
        model.load_state_dict(torch.load("/home/ubuntu/data/home/ubuntu/deeplogic/ql_model_params/epochs0")['model_state_dict'])
        model.eval()
        correct = 0
        tot = 0
        ti=0
        with torch.no_grad():
            t0=time()
            for i in range(100):
                print(i,flush=True)
                valid_loader,X_valid_shape= get_dataset(i,mapping_entities,mapping_properties,max_e,max_p)
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
    mapping_entities={"o3": 0, "o1": 1, "o5": 2, "BrownMetallicSphereLargeObject": 3, "o0": 4, "o2": 5, "GreenMetallicCylinderLargeObject": 6, "o4": 7, "GrayRubberCubeSmallObject": 8, "BrownMetallicCylinderLargeObject": 9, "BlueRubberCubeLargeObject": 10, "CyanRubberCubeSmallObject": 11, "CyanSmallObject": 12, "SizedObject": 13, "MaterialObject": 14, "CubeLargeObject": 15, "ColoredMaterialShapedObject": 16, "RubberObject": 17, "MetallicLargeObject": 18, "ColoredMaterialShapedSizedObject": 19, "GreenObject": 20, "NamedIndividual": 21, "ColoredShapedSizedObject": 22, "RubberCubeLargeObject": 23, "ColoredMaterialSizedObject": 24, "ColoredShapedObject": 25, "ColoredObject": 26, "MaterialSizedObject": 27, "MetallicObject": 28, "CubeObject": 29, "GreenMetallicObject": 30, "RubberCubeSmallObject": 31, "BlueCubeObject": 32, "SmallObject": 33, "CyanCubeSmallObject": 34, "LargeObject": 35, "BrownMetallicObject": 36, "BrownObject": 37, "ShapedObject": 38, "BrownSphereObject": 39, "Thing": 40, "GraySmallObject": 41, "CylinderObject": 42, "BrownMetallicSphereObject": 43, "BlueRubberLargeObject": 44, "BlueRubberObject": 45, "MaterialShapedSizedObject": 46, "CubeSmallObject": 47, "GrayCubeSmallObject": 48, "GreenMetallicLargeObject": 49, "GrayRubberObject": 50, "BlueLargeObject": 51, "CyanObject": 52, "MetallicSphereObject": 53, "BlueObject": 54, "RubberCubeObject": 55, "ShapedSizedObject": 56, "BrownCylinderObject": 57, "ColoredMaterialObject": 58, "MaterialShapedObject": 59, "CyanCubeObject": 60, "BrownMetallicLargeObject": 61, "CyanRubberCubeObject": 62, "BrownLargeObject": 63, "MetallicSphereLargeObject": 64, "Object": 65, "ColoredSizedObject": 66, "RubberSmallObject": 67, "CyanRubberObject": 68, "GrayObject": 69, "BlueCubeLargeObject": 70, "GrayRubberCubeObject": 71, "SphereLargeObject": 72, "BrownSphereLargeObject": 73, "CylinderLargeObject": 74, "RubberLargeObject": 75, "GrayCubeObject": 76, "SphereObject": 77, "CyanRubberSmallObject": 78, "BrownMetallicCylinderObject": 79, "ClevrQL_inferred": 80, "Ontology": 81, "BrownCylinderLargeObject": 82, "GreenLargeObject": 83, "MetallicCylinderObject": 84, "MetallicCylinderLargeObject": 85, "GrayRubberSmallObject": 86, "BlueRubberCubeObject": 87, "GreenCylinderLargeObject": 88, "GreenCylinderObject": 89, "GreenMetallicCylinderObject": 90, "o7": 91, "o8": 92, "BrownRubberCubeSmallObject": 93, "RedRubberSphereSmallObject": 94, "GrayMetallicSphereSmallObject": 95, "o6": 96, "GrayRubberCubeLargeObject": 97, "YellowRubberCubeSmallObject": 98, "BlueRubberSphereLargeObject": 99, "YellowMetallicSphereSmallObject": 100, "GreenRubberSphereSmallObject": 101, "YellowRubberCylinderSmallObject": 102, "SphereSmallObject": 103, "MetallicSphereSmallObject": 104, "BrownRubberObject": 105, "YellowSmallObject": 106, "MetallicSmallObject": 107, "YellowCylinderSmallObject": 108, "RedRubberObject": 109, "BrownCubeObject": 110, "YellowCylinderObject": 111, "YellowMetallicSphereObject": 112, "GreenRubberSphereObject": 113, "RubberCylinderObject": 114, "CylinderSmallObject": 115, "GraySphereSmallObject": 116, "BrownSmallObject": 117, "RubberSphereObject": 118, "GreenRubberSmallObject": 119, "RedObject": 120, "BlueSphereObject": 121, "GrayMetallicSmallObject": 122, "YellowRubberCubeObject": 123, "GreenSphereSmallObject": 124, "YellowRubberCylinderObject": 125, "YellowRubberObject": 126, "YellowRubberSmallObject": 127, "GrayLargeObject": 128, "GreenRubberObject": 129, "RubberSphereSmallObject": 130, "BrownRubberSmallObject": 131, "YellowObject": 132, "BrownCubeSmallObject": 133, "GrayMetallicObject": 134, "YellowMetallicSmallObject": 135, "GrayRubberLargeObject": 136, "GrayCubeLargeObject": 137, "RedRubberSphereObject": 138, "GrayMetallicSphereObject": 139, "BrownRubberCubeObject": 140, "RubberSphereLargeObject": 141, "YellowSphereSmallObject": 142, "RedSphereSmallObject": 143, "GreenSphereObject": 144, "YellowCubeObject": 145, "RedSphereObject": 146, "YellowCubeSmallObject": 147, "GreenSmallObject": 148, "RedRubberSmallObject": 149, "RubberCylinderSmallObject": 150, "YellowMetallicObject": 151, "YellowSphereObject": 152, "BlueSphereLargeObject": 153, "GraySphereObject": 154, "RedSmallObject": 155, "BlueRubberSphereObject": 156, "YellowMetallicSphereLargeObject": 157, "YellowMetallicLargeObject": 158, "YellowLargeObject": 159, "YellowSphereLargeObject": 160, "CyanMetallicCylinderLargeObject": 161, "o9": 162, "GreenRubberSphereLargeObject": 163, "CyanMetallicSphereLargeObject": 164, "PurpleRubberCubeSmallObject": 165, "GrayMetallicCylinderLargeObject": 166, "BlueMetallicCubeSmallObject": 167, "CyanLargeObject": 168, "CyanMetallicObject": 169, "GrayCylinderObject": 170, "BlueMetallicCubeObject": 171, "CyanCylinderObject": 172, "PurpleCubeObject": 173, "PurpleSmallObject": 174, "PurpleRubberCubeObject": 175, "GrayMetallicCylinderObject": 176, "CyanMetallicCylinderObject": 177, "CyanMetallicSphereObject": 178, "GrayMetallicLargeObject": 179, "PurpleCubeSmallObject": 180, "BlueMetallicObject": 181, "PurpleRubberObject": 182, "PurpleRubberSmallObject": 183, "GreenSphereLargeObject": 184, "CyanMetallicLargeObject": 185, "GrayCylinderLargeObject": 186, "BlueMetallicSmallObject": 187, "MetallicCubeObject": 188, "BlueSmallObject": 189, "CyanCylinderLargeObject": 190, "CyanSphereObject": 191, "PurpleObject": 192, "CyanSphereLargeObject": 193, "GreenRubberLargeObject": 194, "MetallicCubeSmallObject": 195, "BlueCubeSmallObject": 196, "CyanMetallicCylinderSmallObject": 197, "PurpleMetallicSphereSmallObject": 198, "PurpleMetallicObject": 199, "PurpleMetallicSphereObject": 200, "PurpleMetallicSmallObject": 201, "PurpleSphereObject": 202, "CyanMetallicSmallObject": 203, "CyanCylinderSmallObject": 204, "PurpleSphereSmallObject": 205, "MetallicCylinderSmallObject": 206, "BrownRubberCylinderLargeObject": 207, "YellowRubberSphereSmallObject": 208, "BlueRubberCylinderLargeObject": 209, "YellowRubberCubeLargeObject": 210, "GreenMetallicCubeLargeObject": 211, "PurpleRubberCubeLargeObject": 212, "RedMetallicCubeSmallObject": 213, "GrayMetallicSphereLargeObject": 214, "CyanRubberSphereSmallObject": 215, "YellowRubberLargeObject": 216, "RubberCylinderLargeObject": 217, "BlueCylinderObject": 218, "GreenCubeLargeObject": 219, "RedCubeObject": 220, "PurpleRubberLargeObject": 221, "CyanSphereSmallObject": 222, "RedCubeSmallObject": 223, "RedMetallicObject": 224, "YellowRubberSphereObject": 225, "PurpleLargeObject": 226, "GreenMetallicCubeObject": 227, "GraySphereLargeObject": 228, "RedMetallicSmallObject": 229, "YellowCubeLargeObject": 230, "CyanRubberSphereObject": 231, "RedMetallicCubeObject": 232, "GreenCubeObject": 233, "BlueCylinderLargeObject": 234, "MetallicCubeLargeObject": 235, "PurpleCubeLargeObject": 236, "BrownRubberCylinderObject": 237, "BlueRubberCylinderObject": 238, "BrownRubberLargeObject": 239, "GreenRubberCubeSmallObject": 240, "GreenRubberCubeObject": 241, "GreenCubeSmallObject": 242, "PurpleMetallicCubeLargeObject": 243, "GreenMetallicCubeSmallObject": 244, "BlueRubberCubeSmallObject": 245, "BlueRubberSphereSmallObject": 246, "GreenMetallicSmallObject": 247, "BlueRubberSmallObject": 248, "PurpleMetallicCubeObject": 249, "PurpleMetallicLargeObject": 250, "BlueSphereSmallObject": 251, "BrownRubberSphereSmallObject": 252, "BrownRubberSphereObject": 253, "BrownSphereSmallObject": 254, "CyanRubberCylinderSmallObject": 255, "YellowMetallicCylinderSmallObject": 256, "GrayRubberCylinderSmallObject": 257, "CyanRubberSphereLargeObject": 258, "GrayCylinderSmallObject": 259, "YellowMetallicCylinderObject": 260, "GrayRubberCylinderObject": 261, "CyanRubberCylinderObject": 262, "CyanRubberLargeObject": 263, "RedMetallicCubeLargeObject": 264, "RedMetallicCylinderSmallObject": 265, "RedMetallicSphereLargeObject": 266, "GreenRubberCylinderSmallObject": 267, "GreenMetallicCylinderSmallObject": 268, "RedCylinderSmallObject": 269, "RedLargeObject": 270, "RedMetallicLargeObject": 271, "RedSphereLargeObject": 272, "RedCubeLargeObject": 273, "RedCylinderObject": 274, "GreenCylinderSmallObject": 275, "RedMetallicCylinderObject": 276, "GreenRubberCylinderObject": 277, "RedMetallicSphereObject": 278, "BlueMetallicCylinderLargeObject": 279, "GreenMetallicSphereLargeObject": 280, "PurpleRubberCylinderLargeObject": 281, "PurpleMetallicCylinderSmallObject": 282, "PurpleMetallicSphereLargeObject": 283, "CyanMetallicCubeSmallObject": 284, "GreenMetallicSphereObject": 285, "BlueMetallicLargeObject": 286, "PurpleMetallicCylinderObject": 287, "PurpleCylinderSmallObject": 288, "BlueMetallicCylinderObject": 289, "PurpleCylinderObject": 290, "PurpleRubberCylinderObject": 291, "PurpleCylinderLargeObject": 292, "CyanMetallicCubeObject": 293, "PurpleSphereLargeObject": 294, "BlueMetallicCylinderSmallObject": 295, "BlueCylinderSmallObject": 296, "RedMetallicCylinderLargeObject": 297, "YellowMetallicCylinderLargeObject": 298, "RedCylinderLargeObject": 299, "YellowCylinderLargeObject": 300, "RedRubberCubeSmallObject": 301, "GreenMetallicSphereSmallObject": 302, "RedRubberCubeObject": 303, "BrownRubberSphereLargeObject": 304, "GrayMetallicCylinderSmallObject": 305, "BrownMetallicCubeLargeObject": 306, "BrownMetallicSphereSmallObject": 307, "YellowRubberCylinderLargeObject": 308, "GrayMetallicCubeLargeObject": 309, "BrownCubeLargeObject": 310, "BrownMetallicCubeObject": 311, "BrownMetallicSmallObject": 312, "GrayMetallicCubeObject": 313, "BlueMetallicSphereLargeObject": 314, "PurpleMetallicCylinderLargeObject": 315, "BlueMetallicSphereObject": 316, "CyanMetallicCubeLargeObject": 317, "GreenRubberCubeLargeObject": 318, "YellowMetallicCubeSmallObject": 319, "RedMetallicSphereSmallObject": 320, "CyanCubeLargeObject": 321, "YellowMetallicCubeObject": 322, "GrayRubberSphereSmallObject": 323, "PurpleMetallicCubeSmallObject": 324, "GrayRubberSphereObject": 325, "BlueMetallicSphereSmallObject": 326, "GrayRubberCylinderLargeObject": 327, "PurpleRubberCylinderSmallObject": 328, "RedRubberSphereLargeObject": 329, "RedRubberLargeObject": 330, "BlueRubberCylinderSmallObject": 331, "RedRubberCubeLargeObject": 332, "CyanMetallicSphereSmallObject": 333, "PurpleRubberSphereSmallObject": 334, "PurpleRubberSphereObject": 335, "RedRubberCylinderSmallObject": 336, "YellowRubberSphereLargeObject": 337, "RedRubberCylinderObject": 338, "GrayMetallicCubeSmallObject": 339, "YellowMetallicCubeLargeObject": 340, "GreenRubberCylinderLargeObject": 341, "GrayRubberSphereLargeObject": 342, "BrownRubberCylinderSmallObject": 343, "BrownCylinderSmallObject": 344, "BrownMetallicCubeSmallObject": 345, "BlueMetallicCubeLargeObject": 346, "CyanRubberCubeLargeObject": 347, "CyanRubberCylinderLargeObject": 348, "PurpleRubberSphereLargeObject": 349, "BrownMetallicCylinderSmallObject": 350, "BrownRubberCubeLargeObject": 351, "RedRubberCylinderLargeObject": 352}
    mapping_properties={"hasDirectlyOnLeft": 0, "type": 1, "hasDirectlyBehind": 2, "hasDirectlyOnFront": 3, "hasDirectlyOnRight": 4, "hasOnLeft": 5, "hasDirectlyNear": 6, "hasNear": 7, "hasOnRight": 8, "hasBehind": 9, "hasOnFront": 10}
    options=[[9,1,1]]
    test_model(options,353,11,mapping_entities,mapping_properties)
    #ql
    """
     {'n_layers': 1, 'kernel_size_l0': 9}
    """
    #options=[[9,1,1]]
    #train_model(options,353,11)




