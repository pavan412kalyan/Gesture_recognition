# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt 

#heads =['Frames#', 'score_overall', 'nose_score', 'nose_x', 'nose_y', 'leftEye_score', 'leftEye_x', 'leftEye_y', 
# 'rightEye_score', 'rightEye_x', 'rightEye_y', 'leftEar_score', 'leftEar_x', 'leftEar_y', 
# 'rightEar_score', 'rightEar_x', 'rightEar_y', 'leftShoulder_score', 'leftShoulder_x', 
# 'leftShoulder_y', 'rightShoulder_score', 'rightShoulder_x', 'rightShoulder_y', 'leftElbow_score',
# 'leftElbow_x', 'leftElbow_y', 'rightElbow_score', 'rightElbow_x', 'rightElbow_y', 
# 'leftWrist_score', 'leftWrist_x', 'leftWrist_y', 'rightWrist_score', 'rightWrist_x', 
# 'rightWrist_y', 'leftHip_score', 'leftHip_x', 'leftHip_y', 
# 'rightHip_score', 'rightHip_x', 'rightHip_y', 'leftKnee_score', 'leftKnee_x',
# 'leftKnee_y', 'rightKnee_score', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 
# 'leftAnkle_x', 'leftAnkle_y', 'rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y']
def normalization(csvdata) :    
    norm_data = []
    for  k in range(len(csvdata)) :
        norm_rw__x_data =[]
        norm_rw__y_data=[]
        norm_lw__x_data=[]
        norm_lw__y_data=[]
        norm_re__y_data=[]
        norm_le__y_data=[]
        norm_re__x_data=[]
        norm_le__x_data=[]
        for i in range(len(csvdata[k])) :
            rw_x =csvdata[k]['rightWrist_x'][i]
            n_x =csvdata[k]['nose_x'][i]
            re_x=csvdata[k]['rightEye_x'][i]
            le_x=csvdata[k]['leftEye_x'][i]
            
            norm_rw_x = rw_x - n_x / (le_x - re_x)
            norm_rw__x_data.append(norm_rw_x)
    
            rw_y =csvdata[k]['rightWrist_y'][i]
            n_y =csvdata[k]['nose_y'][i]
            s_y=csvdata[k]['rightShoulder_y'][i]
                     
            norm_rw_y = rw_y - n_y /n_y - s_y
            norm_rw__y_data.append(norm_rw_y)
            
            lw_x =csvdata[k]['leftWrist_x'][i]
            norm_lw_x = lw_x - n_x / (le_x - re_x)
            norm_lw__x_data.append(norm_lw_x)
            
            lw_y =csvdata[k]['leftWrist_y'][i]
            n_y =csvdata[k]['nose_y'][i]
            s_y=csvdata[k]['rightShoulder_y'][i]
            norm_lw_y = lw_y - n_y /n_y - s_y
            norm_lw__y_data.append(norm_lw_y)
            
            
            re_y =csvdata[k]['rightElbow_y'][i]
            norm_re_y = re_y - n_y /n_y - s_y
            norm_re__y_data.append(norm_re_y)

            le_y =csvdata[k]['leftElbow_y'][i]
            norm_le_y = le_y - n_y /n_y - s_y
            norm_le__y_data.append(norm_le_y)
            
            rE_x =csvdata[k]['rightElbow_x'][i]
            norm_re_x = rE_x - n_x / (le_x - re_x)
            norm_re__x_data.append(norm_re_x)
    

            lE_x =csvdata[k]['leftElbow_x'][i]
            norm_le_x =  lE_x - n_x / (le_x - re_x)
            norm_le__x_data.append(norm_le_x)
                       
            
             
            
            
        norm_data.append(pd.DataFrame(list(zip(norm_rw__x_data, norm_rw__y_data,norm_lw__x_data,norm_lw__y_data,
                                               norm_re__y_data,norm_le__y_data,norm_re__x_data,norm_le__x_data))
            ,columns =['norm_rw__x_data', 'norm_rw__y_data','norm_lw__x_data','norm_lw__y_data','norm_re__y_data',
                       'norm_le__y_data','norm_re__x_data','norm_le__x_data']))
#    normRawData=[]
#    for i in range(len(norm_data)) :
#        rY=norm_data[i]['norm_rw__y_data']
#        normRawData.append((rY - np.mean(rY))/(np.max(rY-np.mean(rY))-np.min(rY-np.mean(rY))))
################################################
    norm_rw__y_moving=[]
    for k in range(len(norm_data)) :
        Y=norm_data[k]['norm_rw__y_data']
        window=9
        Y_new=[]
        for i in range(len(Y)) :
                Y_new.append(sum(Y[i:i+window])/window)
        
        norm_rw__y_moving.append(Y_new)    
    
    
    for i in range(len(norm_rw__y_moving)) :
        norm_data[i]['norm_rw__y_moving']=norm_rw__y_moving[i]
####################################### #########       
    norm_lw__y_moving=[]
    for k in range(len(norm_data)) :
        Y=norm_data[k]['norm_lw__y_data']
        window=9
        Y_new=[]
        for i in range(len(Y)) :
                Y_new.append(sum(Y[i:i+window])/window)
        
        norm_lw__y_moving.append(Y_new)    
            
    for i in range(len(norm_lw__y_moving)) :
        norm_data[i]['norm_lw__y_moving']=norm_lw__y_moving[i]

##########################################  
    norm_rw__x_moving=[]
    for k in range(len(norm_data)) :
        Y=norm_data[k]['norm_rw__x_data']
        window=9
        Y_new=[]
        for i in range(len(Y)) :
                Y_new.append(sum(Y[i:i+window])/window)
        
        norm_rw__x_moving.append(Y_new)    
    
    
    for i in range(len(norm_rw__x_moving)) :
        norm_data[i]['norm_rw__x_moving']=norm_rw__x_moving[i]      
    
#####################################    
    norm_lw__x_moving=[]
    for k in range(len(norm_data)) :
        Y=norm_data[k]['norm_lw__x_data']
        window=9
        Y_new=[]
        for i in range(len(Y)) :
                Y_new.append(sum(Y[i:i+window])/window)
        
        norm_lw__x_moving.append(Y_new)    
    
    
    for i in range(len(norm_lw__x_moving)) :
        norm_data[i]['norm_lw__x_moving']=norm_lw__x_moving[i]      
###########################################################      
    
    norm_re__y_moving=[]
    for k in range(len(norm_data)) :
        Y=norm_data[k]['norm_re__y_data']
        window=9
        Y_new=[]
        for i in range(len(Y)) :
                Y_new.append(sum(Y[i:i+window])/window)
        
        norm_re__y_moving.append(Y_new)    
    
    
    for i in range(len(norm_re__y_moving)) :
        norm_data[i]['norm_re__y_moving']=norm_re__y_moving[i]    


###################################################################

    norm_le__y_moving=[]
    for k in range(len(norm_data)) :
        Y=norm_data[k]['norm_le__y_data']
        window=9
        Y_new=[]
        for i in range(len(Y)) :
                Y_new.append(sum(Y[i:i+window])/window)
        
        norm_le__y_moving.append(Y_new)    
    
    
    for i in range(len(norm_le__y_moving)) :
        norm_data[i]['norm_le__y_moving']=norm_le__y_moving[i] 
##################################

    norm_re__x_moving=[]
    for k in range(len(norm_data)) :
        Y=norm_data[k]['norm_re__x_data']
        window=9
        Y_new=[]
        for i in range(len(Y)) :
                Y_new.append(sum(Y[i:i+window])/window)
        
        norm_re__x_moving.append(Y_new)    
    
    
    for i in range(len(norm_re__x_moving)) :
        norm_data[i]['norm_re__x_moving']=norm_re__x_moving[i]            
##################################################
    norm_le__x_moving=[]
    for k in range(len(norm_data)) :
        Y=norm_data[k]['norm_le__x_data']
        window=9
        Y_new=[]
        for i in range(len(Y)) :
                Y_new.append(sum(Y[i:i+window])/window)
        
        norm_le__x_moving.append(Y_new)    
    
    
    for i in range(len(norm_le__x_moving)) :
        norm_data[i]['norm_le__x_moving']=norm_le__x_moving[i]   
       
#####################################


    
    return norm_data

def max_diff_at_crossing(norm_data,keypoint) :
    window=9    
    max_slope_diff_feature=[]
    for k in range(len(norm_data)) :
        Y=norm_data[k][keypoint]
        Y=Y[0:-window]
        X=[i for i in range(len(Y))] 
        slope=[]
        for i in range(len(X)-1) :
            m=(Y[i+1] - Y[i])/(X[i+1]-X[i])
            slope.append(m)
        #plt.scatter(X[0:30],slope)
        slope_diff=[]
        window_size=5
        for i in range(len(slope)-window_size) :
            mn=min(slope[i:i+window_size])
            mx=max(slope[i:i+window_size])
            slope_diff.append([mx-mn,X[i-1],X[i+window_size]]) 
        max_slope_diff=max(slope_diff)[0]
        max_slope_diff_feature.append(max_slope_diff)
    return max_slope_diff_feature
       
#    print(max(slope_diff)[0])


####################
    
#feature1.2 Zero Crossings
def zerocross(norm_data,keypoint) :
    window=9
#    max_neg_slope_diff=[]
    zero_crossing_count=[]
#    sample=norm_data[i]['norm_rw__y_data']
    
    for k in range(len(norm_data)) :
#        Y=norm_data[k]['norm_rw__y_data']
        Y=norm_data[k][keypoint]
        Y=Y[0:-window]##remove all last 
        X=[i for i in range(len(Y))]
         
        slope=[]
        for j in range(len(X)-1) :
                
            m=(Y[j+1]-Y[j])/(X[j+1]-X[j])
            slope.append(m)
        neg_slope=[]
        z_count=0
        for i in range(len(slope)-2) :
            if slope[i]*slope[i+1] < 0  : 
                neg_slope.append([slope[i+1]-slope[i],X[i+1]])
                z_count=z_count+1;
               
                
        zero_crossing_count.append(z_count)                           

    return zero_crossing_count

##############################
def distance_constant(norm_data,keypoint) :   
    distance=[]    
    for k in range(len(norm_data)) :
#        keypoint='norm_le__y_moving'
        Y=norm_data[k][keypoint] 
        Y=Y[30:31]##remove all last
        distance.append(sum(Y)/len(Y))
    return distance    


###########################
 
    






 