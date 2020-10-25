import pandas as pan
import numpy as np
import os
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import distance
from sklearn.metrics.cluster import homogeneity_score

class Cluster:
    import pandas as pan
    import numpy as np
    import os
    import re
    import random
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.spatial import distance
    from sklearn.metrics.cluster import homogeneity_score
    def cluster(self,path):
#        entries = os.listdir(path)
#        self.entries=entries
        string_data=[]
        true_label=[]
        
        entries = os.listdir(path)
        for entry in entries:
            l1=[]
            entry=path+entry
            with open(entry, 'rb') as f:
                data=f.read().decode(errors="replace")
                data=data.lower()
                data=re.sub(r'[^\w\s]','',data)
                data=" ".join(data.split())
                
        #        regex = re.compile('[^a-zA-Z]')
        #        data=regex.sub(' ', data)
        #        data = ''.join(data)
                string_data.append(data)
            true_label.append(entry[-5:-4])
        #    l1.append(entry[8:-6])
        #    true_label.append(l1)
        #
#        print(true_label)        
        #print(string_data)
        # 
        vectorizer = TfidfVectorizer()
        train=vectorizer.fit_transform(string_data).toarray()
        #centroid=random.choices(train,k=5)
        ##print(train[0])
        #print(centroid)
        
        
        ind=np.random.choice(1725,5)
        center=train[ind]
        
#        center=[]
#        center.append(train[250])
#        center.append(train[450])
#        center.append(train[750])
#        center.append(train[10])
#        center.append(train[1710])
        
        #print(len(center))
        #print(center.shape) 
        # Centroids
        k=0
        pred_label=[]
        b0=[]
        b1=[]
        b2=[]
        b3=[]
        b4=[]
        while(k<20):
            b0.clear()
            b1.clear()
            b2.clear()
            b3.clear()
            b4.clear()
            for i in range(len(train)):
                dis_tem=[]
                for j in range(len(center)):
                    x1=[]
                    ds=distance.euclidean(train[i],center[j])
                    x1.append(ds)
        #            print(j)
                    x1.append(j)
                    dis_tem.append(x1)
                    dis_tem.sort(key=lambda x: (x[0]))
        #        print(dis_tem[0])
                bucket=dis_tem[0][1]
                if(bucket==0):
                    b0.append(train[i])
                if(bucket==1):
                    b1.append(train[i])
                if(bucket==2):
                    b2.append(train[i])
                if(bucket==3):
                    b3.append(train[i])
                if(bucket==4):
                    b4.append(train[i])
            center[0]=np.mean(b0,axis=0)
            center[1]=np.mean(b1,axis=0)
            center[2]=np.mean(b2,axis=0)
            center[3]=np.mean(b3,axis=0)
            center[4]=np.mean(b4,axis=0)
        
        #    print(center)
#            print("cdvd") 
#            print(len(b0))
#            print(len(b1))
#            print(len(b2))
#            print(len(b3))
#            print(len(b4))
            k=k+1
        
        pred_label=[]
        for i in range(len(train)):
            dis_tem1=[]
            for j in range(len(center)):
                x11=[]
                ds1=distance.euclidean(train[i],center[j])
                x11.append(ds1)
                x11.append(j)
                dis_tem1.append(x11)
                dis_tem1.sort(key=lambda x:(x[0]))
        #    print(dis_tem1[0])
            pred_label.append(dis_tem1[0][1])
        
        #print(pred_label)
        for i in range(len(entries)):
            anss[entries[i]]=pred_label[i]
        return anss
#        print(homogeneity_score(true_label, pred_label))
