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
            self.write_csv(self.accuracy_path,accuracy)
            y_true = y_true_batch[i]
            y_pred = y_pred_batch[i]
            tn,fp,fn,tp = 0,0,0,0
            y_true = y_true*0.5 
            tp = ((y_pred-y_true)==0.5).sum().item()
            tn = ((y_pred-y_true)==0).sum().item()
            fp = ((y_pred-y_true)==1).sum().item()
            fn = ((y_pred-y_true)==-0.5).sum().item()
            self.write_csv(self.tp_path,tp)
            self.write_csv(self.fp_path,fp)
            self.write_csv(self.tn_path,tn)
            self.write_csv(self.fn_path,fn)
            for prop in list(self.mapping_p.keys()):
                p = self.mapping_p[prop]
                val=y_pred[:][:][p].eq(y_true[:][:][p].view_as(y_pred[:][:][p])).sum().item()/(self.max_e*self.max_e)
                self.write_csv(self.absolute_path+str(prop)+".csv",val)
                
            precision=tp/(tp+fp)
            self.write_csv(self.precision_path,precision)
            recall=tp/(tp+fn)
            self.write_csv(self.recall_path,recall)
            f1=2*precision*recall/(precision+recall)
            self.write_csv(self.f1_path,f1)
            soundness=(tn+fn+tp)/(tn+fp+fn+tp)
            self.write_csv(self.soundness_path,soundness)
            quality=(tn+tp)/(tn+fp+fn+tp)
            self.write_csv(self.quality_path,quality)
       
    def update_loss(self,loss):
        self.write_csv(self.loss_path,loss)

    def write_csv(path,row):
        with open(path,'a') as fd:
            fd.write(row)

    def update_dict(self,mapping_e,mapping_p):
        self.mapping_p = mapping_p
        self.mapping_e = mapping_e