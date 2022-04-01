import optuna
from optuna.trial import TrialState
import torch 
import torch.nn as nn
import numpy as np
import gc
import random
from rdflib import Graph,Namespace
from os import listdir
from os.path import isfile, join

def getIndex(e,mapping):
    if(e.split('/')[-1].split('#')[-1] in mapping.keys()):
        return mapping.get(e.split('/')[-1].split('#')[-1])
    else:
        print("mapping error")

def getPlainIndex(e,mapping):
    if(e in mapping.keys()):
        return mapping.get(e)
    else:
        print(e)
        print("mapping error")   


def embed(mapping_e, mapping_p, graph,max_e,max_p):
    tensor = np.zeros((386,386,16), dtype=np.int32)
    midx_s,midx_o,midx_p=0,0,0
    n = Namespace("http://example.org/")
    q_n_blank = graph.query('SELECT ?s ?p ?o WHERE {?s ?p ?o filter(!isBlank(?s) && !isBlank(?o))}',initNs={ 'ex': n })
    #for s,p,o in graph.triples((None,None,None)):
    for row in q_n_blank:
        idx_s=getIndex(row.s,mapping_e)
        idx_o=getIndex(row.o,mapping_e)
        idx_p=getIndex(row.p,mapping_p)
        tensor[idx_s][idx_o][idx_p]=1
        if(row.p.split('/')[-1].split('#')[-1]=='type'):
            obj = row.o.split('/')[-1].split('#')[-1]
            if(obj=='BlueObject'):
                tensor[idx_s][getPlainIndex('bBlue',mapping_e)][getPlainIndex('hasColor',mapping_p)]=1
                tensor[getPlainIndex('bBlue',mapping_e)][getPlainIndex('Blue',mapping_e)][getPlainIndex('type',mapping_p)]=1
            if(obj=='BrownObject'):
                tensor[idx_s][getPlainIndex('bBrown',mapping_e)][getPlainIndex('hasColor',mapping_p)]=1
                tensor[getPlainIndex('bBrown',mapping_e)][getPlainIndex('Brown',mapping_e)][getPlainIndex('type',mapping_p)]=1
            if(obj=='CyanObject'):
                tensor[idx_s][getPlainIndex('bCyan',mapping_e)][getPlainIndex('hasColor',mapping_p)]=1
                tensor[getPlainIndex('bCyan',mapping_e)][getPlainIndex('Cyan',mapping_e)][getPlainIndex('type',mapping_p)]=1
            if(obj=='GrayObject'):
                tensor[idx_s][getPlainIndex('bGray',mapping_e)][getPlainIndex('hasColor',mapping_p)]=1
                tensor[getPlainIndex('bGray',mapping_e)][getPlainIndex('Gray',mapping_e)][getPlainIndex('type',mapping_p)]=1
            if(obj=='GreenObject'):
                tensor[idx_s][getPlainIndex('bGreen',mapping_e)][getPlainIndex('hasColor',mapping_p)]=1
                tensor[getPlainIndex('bGreen',mapping_e)][getPlainIndex('Green',mapping_e)][getPlainIndex('type',mapping_p)]=1                
            if(obj=='PurpleObject'):
                tensor[idx_s][getPlainIndex('bPurple',mapping_e)][getPlainIndex('hasColor',mapping_p)]=1
                tensor[getPlainIndex('bPurple',mapping_e)][getPlainIndex('Purple',mapping_e)][getPlainIndex('type',mapping_p)]=1
            if(obj=='RedObject'):
                tensor[idx_s][getPlainIndex('bRed',mapping_e)][getPlainIndex('hasColor',mapping_p)]=1
                tensor[getPlainIndex('bRed',mapping_e)][getPlainIndex('Red',mapping_e)][getPlainIndex('type',mapping_p)]=1
            if(obj=='YellowObject'):
                tensor[idx_s][getPlainIndex('bYellow',mapping_e)][getPlainIndex('hasColor',mapping_p)]=1
                tensor[getPlainIndex('bYellow',mapping_e)][getPlainIndex('Yellow',mapping_e)][getPlainIndex('type',mapping_p)]=1
            if(obj=='SmallObject'):
                tensor[idx_s][getPlainIndex('bSmall',mapping_e)][getPlainIndex('hasSize',mapping_p)]=1
                tensor[getPlainIndex('bSmall',mapping_e)][getPlainIndex('Small',mapping_e)][getPlainIndex('type',mapping_p)]=1
            if(obj=='BigObject'):
                tensor[idx_s][getPlainIndex('bBig',mapping_e)][getPlainIndex('hasSize',mapping_p)]=1
                tensor[getPlainIndex('bBig',mapping_e)][getPlainIndex('Big',mapping_e)][getPlainIndex('type',mapping_p)]=1
            if(obj=='MetalObject'):
                tensor[idx_s][getPlainIndex('bMetal',mapping_e)][getPlainIndex('hasMaterial',mapping_p)]=1
                tensor[getPlainIndex('bMetal',mapping_e)][getPlainIndex('Metal',mapping_e)][getPlainIndex('type',mapping_p)]=1                
            if(obj=='RubberObject'):
                tensor[idx_s][getPlainIndex('bRubber',mapping_e)][getPlainIndex('hasMaterial',mapping_p)]=1
                tensor[getPlainIndex('bRubber',mapping_e)][getPlainIndex('Rubber',mapping_e)][getPlainIndex('type',mapping_p)]=1                
            if(obj=='SphereObject'):
                tensor[idx_s][getPlainIndex('bSphere',mapping_e)][getPlainIndex('hasShape',mapping_p)]=1
                tensor[getPlainIndex('bSphere',mapping_e)][getPlainIndex('Sphere',mapping_e)][getPlainIndex('type',mapping_p)]=1                
            if(obj=='CubeObject'):
                tensor[idx_s][getPlainIndex('bCube',mapping_e)][getPlainIndex('hasShape',mapping_p)]=1
                tensor[getPlainIndex('bCube',mapping_e)][getPlainIndex('Cube',mapping_e)][getPlainIndex('type',mapping_p)]=1                
            if(obj=='CylinderObject'):
                tensor[idx_s][getPlainIndex('bCylinder',mapping_e)][getPlainIndex('hasShape',mapping_p)]=1
                tensor[getPlainIndex('bCylinder',mapping_e)][getPlainIndex('Cylinder',mapping_e)][getPlainIndex('type',mapping_p)]=1                
    return tensor


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


def get_dataset(i,index,mapping_entities,mapping_properties):
    X_train=[]
    Y_train=[]
    max_e,max_p = 386,16
    #counter = 0
    mypath="/home/ubuntu/data/home/ubuntu/deeplogic/src/main/resources/dataset/dataset/train_materialized/rl/"
    onlyfiles = [f.split('.')[0][5:] for f in listdir(mypath) if isfile(join(mypath, f))]
    for k in range(i*100+index*1000,(i+1)*100+index*1000):
    #for filename in os.listdir("/home/ubuntu/data/home/ubuntu/deeplogic/el_dataset/x"):
        j=onlyfiles[k]
        n = Namespace("http://example.org/")
        g1 = Graph()
        g1.bind("ex",n)
        g1.parse("/home/ubuntu/data/home/ubuntu/deeplogic/train_pre/scene{}.ttl".format(j), format="turtle")
        spo_tensor= embed(mapping_entities,mapping_properties,g1,max_e,max_p)
        #counter = counter +1
        X_train.append(spo_tensor)
        g2 = Graph()
        g2.bind("ex",n)
        g2.parse("/home/ubuntu/data/home/ubuntu/deeplogic/src/main/resources/dataset/dataset/train_materialized/rl/scene{}.ttl".format(j),format="turtle")

        g3 = Graph()
        g3.parse("/home/ubuntu/data/home/ubuntu/deeplogic/clevr_rl_materialized.ttl", format="ttl")

        graph = g2 - g3
        graph.bind("ex",n)

        spo_tensor = embed(mapping_entities,mapping_properties,graph,max_e,max_p)
        Y_train.append(spo_tensor)
    train_x = torch.from_numpy(np.array(X_train)).float()
    train_y = torch.from_numpy(np.array(Y_train)).float()
    

    batch_size = 5


    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(train_x,train_y)

    # data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)

    return train_loader,len(X_train)

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
    max_e,max_p = 386,16
    DEVICE = get_device()
    EPOCHS = 1
    BATCHSIZE=5
    criterion = torch.nn.BCELoss()
    mapping_entities={"Sphere": 0, "o7": 1, "o2": 2, "Gray": 3, "o4": 4, "o0": 5, "o5": 6, "ShapedObject": 7, "Yellow": 8, "Brown": 9, "Metal": 10, "CylinderLargeObject": 11, "o3": 12, "Small": 13, "Cylinder": 14, "ShapedSizedObject": 15, "Purple": 16, "o6": 17, "Red": 18, "o1": 19, "NamedIndividual": 20, "Big": 21, "BlueCubeLargeObject": 22, "Cube": 23, "ColoredMaterialShapedObject": 24, "ColoredSizedObject": 25, "MaterialObject": 26, "RubberObject": 27, "GrayCubeObject": 28, "SphereObject": 29, "YellowLargeObject": 30, "RubberSphereObject": 31, "Cyan": 32, "SizedObject": 33, "Object": 34, "BrownMetallicSmallObject": 35, "Green": 36, "ColoredShapedSizedObject": 37, "RubberCylinderObject": 38, "ColoredObject": 39, "MaterialSizedObject": 40, "Blue": 41, "MaterialShapedObject": 42, "BrownObject": 43, "SphereSmallObject": 44, "GrayMetallicObject": 45, "CyanRubberObject": 46, "RubberLargeObject": 47, "CyanRubberSmallObject": 48, "ColoredMaterialSizedObject": 49, "ColoredMaterialShapedSizedObject": 50, "MetallicSmallObject": 51, "BrownCubeSmallObject": 52, "LargeObject": 53, "test": 54, "RedObject": 55, "BlueRubberCubeObject": 56, "GrayLargeObject": 57, "CubeObject": 58, "SmallObject": 59, "RubberCubeLargeObject": 60, "Thing": 61, "GrayRubberCylinderObject": 62, "MaterialShapedSizedObject": 63, "CubeLargeObject": 64, "BlueObject": 65, "RubberCubeObject": 66, "CubeSmallObject": 67, "GrayCubeSmallObject": 68, "Rubber": 69, "RubberSmallObject": 70, "GrayRubberCylinderLargeObject": 71, "BlueCubeObject": 72, "CylinderObject": 73, "GrayObject": 74, "CyanObject": 75, "ColoredMaterialObject": 76, "GrayRubberSmallObject": 77, "GrayCylinderObject": 78, "YellowRubberLargeObject": 79, "GrayRubberSphereSmallObject": 80, "RedRubberObject": 81, "GrayMetallicCubeSmallObject": 82, "BlueRubberObject": 83, "GrayMetallicCubeObject": 84, "CyanRubberSphereObject": 85, "ColoredShapedObject": 86, "GraySmallObject": 87, "BlueRubberLargeObject": 88, "GraySphereObject": 89, "RedSphereSmallObject": 90, "CyanRubberSphereSmallObject": 91, "RubberSphereSmallObject": 92, "GrayRubberSphereObject": 93, "BlueRubberCubeLargeObject": 94, "RedSmallObject": 95, "RedRubberSphereSmallObject": 96, "RedSphereObject": 97, "CyanSmallObject": 98, "RedRubberSphereObject": 99, "MetallicCubeSmallObject": 100, "CyanSphereSmallObject": 101, "GrayRubberObject": 102, "MetallicCubeObject": 103, "RedRubberSmallObject": 104, "YellowRubberCubeLargeObject": 105, "BrownMetallicObject": 106, "BrownMetallicCubeSmallObject": 107, "BlueLargeObject": 108, "BrownCubeObject": 109, "YellowRubberObject": 110, "CyanSphereObject": 111, "GrayCylinderLargeObject": 112, "GrayMetallicSmallObject": 113, "RubberCylinderLargeObject": 114, "GraySphereSmallObject": 115, "YellowRubberCubeObject": 116, "BrownMetallicCubeObject": 117, "MetallicObject": 118, "BrownSmallObject": 119, "YellowCubeLargeObject": 120, "YellowObject": 121, "YellowCubeObject": 122, "GrayRubberLargeObject": 123, "YellowMetallicSmallObject": 124, "YellowSphereObject": 125, "MetallicSphereSmallObject": 126, "YellowMetallicSphereObject": 127, "YellowMetallicObject": 128, "YellowSmallObject": 129, "YellowMetallicSphereSmallObject": 130, "YellowSphereSmallObject": 131, "MetallicSphereObject": 132, "o8": 133, "o9": 134, "GreenRubberCubeObject": 135, "GreenRubberCubeSmallObject": 136, "PurpleMetallicSphereSmallObject": 137, "PurpleMetallicObject": 138, "YellowCylinderLargeObject": 139, "GreenSmallObject": 140, "PurpleMetallicSphereObject": 141, "GrayMetallicCylinderSmallObject": 142, "CylinderSmallObject": 143, "RubberCubeSmallObject": 144, "YellowCylinderSmallObject": 145, "GreenRubberSmallObject": 146, "YellowRubberCylinderSmallObject": 147, "MetallicCylinderSmallObject": 148, "PurpleMetallicSmallObject": 149, "GreenObject": 150, "YellowMetallicLargeObject": 151, "PurpleMetallicCubeSmallObject": 152, "PurpleObject": 153, "PurpleMetallicCylinderLargeObject": 154, "GrayCylinderSmallObject": 155, "MetallicCylinderObject": 156, "YellowRubberCylinderObject": 157, "YellowRubberCubeSmallObject": 158, "YellowRubberSmallObject": 159, "PurpleSmallObject": 160, "YellowMetallicCylinderObject": 161, "PurpleCylinderLargeObject": 162, "PurpleCubeSmallObject": 163, "GreenMetallicSmallObject": 164, "GreenCylinderSmallObject": 165, "PurpleSphereSmallObject": 166, "MetallicCylinderLargeObject": 167, "PurpleMetallicCubeObject": 168, "YellowCubeSmallObject": 169, "GreenCubeSmallObject": 170, "GrayMetallicCylinderObject": 171, "YellowCylinderObject": 172, "GreenCylinderObject": 173, "PurpleMetallicLargeObject": 174, "GreenMetallicCylinderSmallObject": 175, "PurpleLargeObject": 176, "GreenRubberObject": 177, "RubberCylinderSmallObject": 178, "GreenCubeObject": 179, "MetallicLargeObject": 180, "YellowMetallicCylinderLargeObject": 181, "GreenMetallicObject": 182, "PurpleMetallicCylinderObject": 183, "PurpleCubeObject": 184, "PurpleSphereObject": 185, "PurpleCylinderObject": 186, "GreenMetallicCylinderObject": 187, "BlueSphereSmallObject": 188, "GreenMetallicCubeSmallObject": 189, "BlueRubberSphereObject": 190, "PurpleRubberObject": 191, "PurpleRubberCubeLargeObject": 192, "PurpleCubeLargeObject": 193, "PurpleRubberLargeObject": 194, "BlueSphereObject": 195, "GreenMetallicCubeObject": 196, "BlueRubberSphereSmallObject": 197, "RubberSphereLargeObject": 198, "SphereLargeObject": 199, "BlueRubberSmallObject": 200, "CyanSphereLargeObject": 201, "BlueRubberSphereLargeObject": 202, "BlueSmallObject": 203, "CyanRubberLargeObject": 204, "PurpleRubberCubeObject": 205, "CyanLargeObject": 206, "CyanRubberSphereLargeObject": 207, "BlueSphereLargeObject": 208, "BrownMetallicSphereSmallObject": 209, "BlueRubberCubeSmallObject": 210, "BrownSphereObject": 211, "GreenRubberSphereObject": 212, "GreenSphereObject": 213, "BrownSphereSmallObject": 214, "GreenSphereSmallObject": 215, "GreenRubberSphereSmallObject": 216, "BrownMetallicSphereObject": 217, "BlueCubeSmallObject": 218, "PurpleCylinderSmallObject": 219, "PurpleRubberCylinderObject": 220, "PurpleRubberSmallObject": 221, "PurpleRubberCylinderSmallObject": 222, "GreenMetallicLargeObject": 223, "BrownRubberObject": 224, "GreenLargeObject": 225, "CyanMetallicCubeSmallObject": 226, "CyanCubeObject": 227, "GreenSphereLargeObject": 228, "BrownRubberCubeObject": 229, "GreenMetallicSphereLargeObject": 230, "CyanMetallicObject": 231, "CyanMetallicCubeObject": 232, "BrownRubberSmallObject": 233, "CyanCubeSmallObject": 234, "CyanMetallicSmallObject": 235, "MetallicSphereLargeObject": 236, "BrownRubberCubeSmallObject": 237, "GreenMetallicSphereObject": 238, "GreenRubberCylinderLargeObject": 239, "GreenRubberCylinderObject": 240, "GreenRubberLargeObject": 241, "BrownRubberCylinderObject": 242, "BrownRubberCylinderLargeObject": 243, "BrownCylinderObject": 244, "PurpleRubberCubeSmallObject": 245, "BrownRubberLargeObject": 246, "BrownLargeObject": 247, "BrownCylinderLargeObject": 248, "GreenCylinderLargeObject": 249, "RedMetallicSmallObject": 250, "RedMetallicCubeSmallObject": 251, "RedMetallicCubeObject": 252, "RedCubeSmallObject": 253, "RedMetallicObject": 254, "RedCubeObject": 255, "PurpleMetallicCylinderSmallObject": 256, "GrayMetallicSphereObject": 257, "GreenRubberSphereLargeObject": 258, "GrayMetallicSphereSmallObject": 259, "CyanRubberCylinderObject": 260, "CyanRubberCylinderSmallObject": 261, "CyanCylinderSmallObject": 262, "CyanCylinderObject": 263, "RedCylinderLargeObject": 264, "RedLargeObject": 265, "RedRubberCylinderObject": 266, "RedRubberSphereLargeObject": 267, "GrayMetallicCylinderLargeObject": 268, "RedCylinderObject": 269, "RedMetallicCubeLargeObject": 270, "RedRubberLargeObject": 271, "RedSphereLargeObject": 272, "RedRubberCylinderLargeObject": 273, "RedMetallicLargeObject": 274, "MetallicCubeLargeObject": 275, "GrayMetallicLargeObject": 276, "RedCubeLargeObject": 277, "YellowMetallicSphereLargeObject": 278, "YellowMetallicCubeLargeObject": 279, "YellowMetallicCubeObject": 280, "YellowSphereLargeObject": 281, "BrownMetallicLargeObject": 282, "RedMetallicCylinderSmallObject": 283, "BrownMetallicCubeLargeObject": 284, "BrownCubeLargeObject": 285, "CyanMetallicCylinderLargeObject": 286, "RedCylinderSmallObject": 287, "CyanMetallicLargeObject": 288, "RedMetallicCylinderObject": 289, "CyanMetallicCylinderObject": 290, "CyanCylinderLargeObject": 291, "RedRubberCubeSmallObject": 292, "BrownRubberSphereObject": 293, "YellowMetallicCylinderSmallObject": 294, "BrownSphereLargeObject": 295, "RedRubberCubeObject": 296, "CyanCubeLargeObject": 297, "BrownRubberSphereLargeObject": 298, "CyanMetallicCubeLargeObject": 299, "GrayRubberSphereLargeObject": 300, "GraySphereLargeObject": 301, "BrownRubberSphereSmallObject": 302, "BlueMetallicLargeObject": 303, "PurpleRubberCylinderLargeObject": 304, "GreenCubeLargeObject": 305, "GreenMetallicCubeLargeObject": 306, "BlueMetallicObject": 307, "BlueMetallicCubeObject": 308, "BlueMetallicCubeLargeObject": 309, "CyanMetallicCylinderSmallObject": 310, "BlueMetallicCylinderSmallObject": 311, "BlueCylinderSmallObject": 312, "BlueMetallicCylinderObject": 313, "BlueCylinderObject": 314, "BlueMetallicSmallObject": 315, "GrayCubeLargeObject": 316, "BlueRubberCylinderObject": 317, "CyanRubberCubeObject": 318, "CyanRubberCubeSmallObject": 319, "GrayRubberCubeLargeObject": 320, "GrayRubberCubeObject": 321, "BlueRubberCylinderLargeObject": 322, "BlueCylinderLargeObject": 323, "YellowRubberCylinderLargeObject": 324, "RedMetallicSphereObject": 325, "YellowRubberSphereSmallObject": 326, "RedMetallicSphereLargeObject": 327, "YellowRubberSphereObject": 328, "PurpleRubberSphereObject": 329, "PurpleRubberSphereSmallObject": 330, "RedMetallicCylinderLargeObject": 331, "GrayRubberCubeSmallObject": 332, "GreenRubberCylinderSmallObject": 333, "GrayMetallicSphereLargeObject": 334, "CyanMetallicSphereObject": 335, "CyanMetallicSphereLargeObject": 336, "YellowRubberSphereLargeObject": 337, "BrownMetallicCylinderObject": 338, "BrownMetallicCylinderSmallObject": 339, "BlueMetallicCubeSmallObject": 340, "CyanRubberCylinderLargeObject": 341, "RedRubberCylinderSmallObject": 342, "BrownCylinderSmallObject": 343, "BlueMetallicCylinderLargeObject": 344, "BrownRubberCubeLargeObject": 345, "BrownMetallicSphereLargeObject": 346, "RedRubberCubeLargeObject": 347, "PurpleRubberSphereLargeObject": 348, "BrownMetallicCylinderLargeObject": 349, "PurpleSphereLargeObject": 350, "PurpleMetallicCubeLargeObject": 351, "GreenRubberCubeLargeObject": 352, "GreenMetallicCylinderLargeObject": 353, "BrownRubberCylinderSmallObject": 354, "GreenMetallicSphereSmallObject": 355, "BlueMetallicSphereObject": 356, "YellowMetallicCubeSmallObject": 357, "BlueMetallicSphereLargeObject": 358, "GrayRubberCylinderSmallObject": 359, "RedMetallicSphereSmallObject": 360, "CyanMetallicSphereSmallObject": 361, "BlueRubberCylinderSmallObject": 362, "PurpleMetallicSphereLargeObject": 363, "CyanRubberCubeLargeObject": 364, "GrayMetallicCubeLargeObject": 365, "BlueMetallicSphereSmallObject": 366,"Color":367,"Shape":368, "Size":369, "Material":370,"bBlue":371,"bBrown":372,"bCyan":373,"bGray":374,"bGreen":375,"bPurple":376,"bRed":377,"bYellow":378,"bBig":379,"bSmall":380,"bMetal":381,"bRubber":382,"bCube":383,"bSphere":384,"bCylinder":385}
    mapping_properties={"topObjectProperty": 0, "hasBehind": 1, "type": 2, "hasNear": 3, "hasDirectlyOnFront": 4, "hasOnLeft": 5, "hasDirectlyOnLeft": 6, "hasOnRight": 7, "hasOnFront": 8, "hasDirectlyOnRight": 9, "hasDirectlyNear": 10, "hasDirectlyBehind": 11,"hasColor":12,"hasShape":13,"hasMaterial":14,"hasSize":15}
    try: 
        # Generate the model.
        model = AutoEncoder(trial,max_e,max_p).to(DEVICE)
        #print(model)
        # Generate the optimizers.
        lr = 0.09
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

        index = random.randint(0,5)
        # Training of the model.
        for epoch in range(EPOCHS):
            model.train()
            for i in range(8):
                train_loader,X_train_shape= get_dataset(i,index,mapping_entities,mapping_properties)
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
                    valid_loader,X_valid_shape= get_dataset(i,index,mapping_entities,mapping_properties)
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


