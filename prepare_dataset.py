from rdflib import Graph,Namespace
import numpy as np
import torch
import argparse

def getIndex(e,mapping):
    if(e.split('/')[-1].split('#')[-1] in mapping.keys()):
        return mapping.get(e.split('/')[-1].split('#')[-1]),mapping
    if(e.split('/')[-1].split('#')[-1] not in mapping.keys()):
        mapping[e.split('/')[-1].split('#')[-1]]=len(mapping.keys())
        return mapping.get(e.split('/')[-1].split('#')[-1]),mapping
import os
def get_maximum_sizes(folder_path, ontology_path):
    max_e=0
    max_p=0
    counter =0 
    for filename in os.listdir(folder_path):
        print(counter)
        counter+=1
        if filename.endswith(".ttl"):
            #print(filename)
            try:
                n = Namespace("http://example.org/")
                g1 = Graph()
                g1.parse(folder_path+'/'+filename, format="ttl")
                
                g2 = Graph()
                g2.parse(ontology_path, format="ttl")

                graph = g1 - g2
                graph.bind("ex",n)
                qs=graph.query('SELECT ?s ?p ?o WHERE { ?s ?p ?o }',initNs={ 'ex': n })
                subjects = [row.s for row in qs]
                objects = [row.o for row in qs]
                properties = [row.p for row in qs]

                unique_subjects = {s for s in subjects}
                unique_objects = {o for o in objects}
                unique_properties = {p for p in properties}
                unique_entities = unique_objects.union(unique_subjects)

                if(len(unique_properties)>max_p):
                    max_p = len(unique_properties)
                if(len(unique_entities)>max_e):
                    max_e =len(unique_entities)
            except Exception as e:
                print(filename)
                
    print(max_e,max_p)
    return max_e,max_p



def embed(mapping_e, mapping_p, graph,max_e,max_p):
    tensor = np.zeros((max_e,max_e,max_p), dtype=np.int32)
    midx_s,midx_o,midx_p=0,0,0
    a,b,c='','',''
    try:
        for s,p,o in graph.triples((None,None,None)):
            idx_s,mapping_e=getIndex(s,mapping_e)
            idx_o,mapping_e=getIndex(o,mapping_e)
            idx_p,mapping_p=getIndex(p,mapping_p)
            a = s
            b=p
            c=o
            if(idx_s>midx_s):
                midx_s=idx_s
            if(idx_o>midx_o):
                midx_o=idx_o
            if(idx_p>midx_p):
                midx_p=idx_p
       
        tensor[idx_s][idx_o][idx_p]=1
    except Exception as e:
        print(e)
        print(a)
        print(b)
        print(c)
    return tensor, mapping_e,mapping_p,midx_s,midx_o,midx_p


def ttl_to_tensor(folder_path, ontology_path, target_folder):
    mapping_entities = {}
    mapping_properties = {}
    #max_e,max_p = get_maximum_sizes(folder_path,ontology_path)
    max_e,max_p = 5000,100
    i_m_e,i_m_p=0,0
    for filename in os.listdir(folder_path):
        if filename.endswith(".ttl"):
            try:
                n = Namespace("http://example.org/")
                g1 = Graph()
                g1.parse(folder_path+'/'+filename, format="ttl")

                g2 = Graph()
                g2.parse(ontology_path, format="ttl")

                graph = g1 - g2
                graph.bind("ex",n)
                #qs=graph.query('SELECT ?s ?p ?o WHERE { ?s ?p ?o }',initNs={ 'ex': n })
                spo_tensor,mapping_entities,mapping_properties,idx_s,idx_o,idx_p = embed(mapping_entities,mapping_properties,graph,max_e,max_p)
                #torch.save(spo_tensor, target_folder+'/'+os.path.splitext(os.path.basename(filename))[0]+'.pt')
                if(idx_s>i_m_e):
                    i_m_e=idx_s
                if(idx_o>i_m_e):
                    i_m_e=idx_o
                if(idx_p>i_m_p):
                    i_m_p=idx_p
            except Exception as e:
                print(filename)
                print(e)
                break
    print(i_m_e)
    print(i_m_p)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder')
    parser.add_argument('--ontology')
    parser.add_argument('--target')
    args = parser.parse_args()

    for arg in args.__dict__.keys():
       if args.__dict__.get(arg) is None:
           sys.exit("Missing argument:"+arg)
    ttl_to_tensor(args.folder, args.ontology, args.target)
