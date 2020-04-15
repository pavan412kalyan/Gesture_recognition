# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt 

from features import normalization,max_diff_at_crossing,zerocross,distance_constant
import pickle  

#['communicate','fun','hope','mother','really']


def combined_features(filelist,label):
    class_v=0
    final_dataset=[]
    for file in  filelist :
        print(file)
        files = os.listdir(file)
        csvdata_buy=[]
        for cs in files :
#            print(cs)
            csvdata_buy.append(pd.read_csv(file+'/'+ cs))
              
        buy_norm=normalization(csvdata_buy) 
               
        dataset=pd.DataFrame()
        class_label=np.full(len(csvdata_buy),class_v)
        dataset['zerocross_diff_rw_y']= max_diff_at_crossing(buy_norm,'norm_rw__y_moving')
        dataset['zero_crossing_count_rw_y']= zerocross(buy_norm,'norm_rw__y_moving')
        
        dataset['zerocross_diff_lw_y']= max_diff_at_crossing(buy_norm,'norm_lw__y_moving')
        dataset['zero_crossing_count_lw_y']= zerocross(buy_norm,'norm_lw__y_moving')
        
        dataset['zerocross_diff_lw_x']= max_diff_at_crossing(buy_norm,'norm_lw__x_moving')
        dataset['zero_crossing_count_lw_x']= zerocross(buy_norm,'norm_lw__x_moving')
        
        dataset['zerocross_diff_rw_x']= max_diff_at_crossing(buy_norm,'norm_rw__x_moving')
        dataset['zero_crossing_count_rw_x']= zerocross(buy_norm,'norm_rw__x_moving')
        
        dataset['zerocross_diff_le_y']= max_diff_at_crossing(buy_norm,'norm_le__y_moving')
        dataset['zero_crossing_count_le_y']= zerocross(buy_norm,'norm_le__y_moving')
        
        dataset['zerocross_diff_re_y']= max_diff_at_crossing(buy_norm,'norm_re__y_moving')
        dataset['zero_crossing_count_re_y']= zerocross(buy_norm,'norm_re__y_moving')


        
        dataset['left_wrist_dis_y']= distance_constant(buy_norm,'norm_lw__y_moving')
        dataset['right_wrist_dis_y']= distance_constant(buy_norm,'norm_rw__y_moving')
        dataset['left_elbow_dis_y']= distance_constant(buy_norm,'norm_le__y_moving')
        dataset['right_elbow_dis_y']= distance_constant(buy_norm,'norm_re__y_moving')
        dataset['left_elbow_dis_x']= distance_constant(buy_norm,'norm_le__x_moving')
        dataset['right_elbow_dis_x']= distance_constant(buy_norm,'norm_re__x_moving')
        dataset['left_wrist_dis_x']= distance_constant(buy_norm,'norm_lw__x_moving')
        dataset['right_wrist_dis_x']= distance_constant(buy_norm,'norm_rw__x_moving')  
        
       
        
        if(label==1) :
            dataset['class_label']=class_label
        final_dataset.append(dataset)
        class_v=class_v+1
    
    mod=pd.concat(final_dataset)
    return mod
######################################



#test_data=combined_features(['test_buy'],0)

model_data=combined_features(['buy','communicate','fun','hope','mother','really'],1)

from sklearn.model_selection import train_test_split
X=model_data.iloc[:,:-1]
y=model_data.iloc[:,-1]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)
X_s=scaler.transform(X)


size=0.001
#####################
X_train, X_test, y_train, y_test = train_test_split( X_s, y, test_size=size, random_state=4)
from sklearn.neural_network import MLPClassifier
ann = MLPClassifier(solver='lbfgs', alpha=0.00001,activation='tanh',
                        hidden_layer_sizes=(15,5),random_state=1)
ann.fit(X_train, y_train)
print(ann.score(X_test,y_test))

from sklearn.metrics import classification_report
y_true = y_test
y_pred = ann.predict(X_test)
#ann.predict(test_data)

print(classification_report(y_true,y_pred))

pkl_filename = "model1_ann.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(ann, file)



#############################

from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=size, random_state=4)

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)
print(knn.score(X_test,y_test))


print(classification_report(y_true,y_pred))


pkl_filename = "model2_knn.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(knn, file)



######################################
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=size, random_state=4)

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
print(qda.score(X_test,y_test))
y_true = y_test
y_pred = qda.predict(X_test)
#qda.predict(test_data)
pkl_filename = "model3_qda.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(qda, file)

print(classification_report(y_true,y_pred))

##################################
from sklearn.linear_model import RidgeClassifierCV
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=size, random_state=4)




rccv = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X_train,y_train)
print(rccv.score(X_test,y_test))
y_true = y_test
y_pred = rccv.predict(X_test)
#rccv.predict(test_data)
pkl_filename = "model4_rccv.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(rccv, file)

print(classification_report(y_true,y_pred))






