# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt 

from features import normalization,max_diff_at_crossing,zerocross,distance_constant
import pickle  
import sys
from flask import Flask, render_template, request, redirect, Response, jsonify

#['communicate','fun','hope','mother','really']

with open("model1_ann.pkl", 'rb') as file:
     ann = pickle.load(file)
with open("model2_knn.pkl", 'rb') as file:
     knn = pickle.load(file)    
with open("model3_qda.pkl", 'rb') as file:
     qda = pickle.load(file)     
with open("model4_rccv.pkl", 'rb') as file:
     rccv= pickle.load(file)
def single_test_features(csvdata_buy):
    
    
    buy_norm=normalization(csvdata_buy) 
               
    dataset=pd.DataFrame()
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
        
        
                  
    return dataset

#csvdata=[] ## intilaze this as list...normalization function expecting as csvdata list.....
#singletest=pd.read_csv('COMMUNICATE_1_DOKE.csv')
#csvdata.append(singletest)
#test=single_test_features(csvdata)
##['buy','communicate','fun','hope','mother','really']
#
#
#gestures[ann.predict(test)[0]]   
#gestures[knn.predict(test)[0]]   
#gestures[qda.predict(test)[0]]   
#gestures[rccv.predict(test)[0]]   


gestures = {0:'buy', 1:'communicate', 2:'fun', 3:'hope', 4:'mother', 5:'really'}

#
columns =['score_overall', 'nose_score', 'nose_x', 'nose_y',
   'leftEye_score', 'leftEye_x', 'leftEye_y', 'rightEye_score',
   'rightEye_x', 'rightEye_y', 'leftEar_score', 'leftEar_x', 'leftEar_y',
   'rightEar_score', 'rightEar_x', 'rightEar_y', 'leftShoulder_score',
   'leftShoulder_x', 'leftShoulder_y', 'rightShoulder_score',
   'rightShoulder_x', 'rightShoulder_y', 'leftElbow_score', 'leftElbow_x',
   'leftElbow_y', 'rightElbow_score', 'rightElbow_x', 'rightElbow_y',
   'leftWrist_score', 'leftWrist_x', 'leftWrist_y', 'rightWrist_score',
   'rightWrist_x', 'rightWrist_y', 'leftHip_score', 'leftHip_x',
   'leftHip_y', 'rightHip_score', 'rightHip_x', 'rightHip_y',
   'leftKnee_score', 'leftKnee_x', 'leftKnee_y', 'rightKnee_score',
   'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 'leftAnkle_x',
   'leftAnkle_y', 'rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y']


def convert_to_dataframe(packet):
    all_frames = []
    for i in range(len(packet)):
        temp = []
        current_frame = packet[i]
        temp.append(current_frame['score'])
        key_points = current_frame['keypoints']
        for j in range(len(key_points)):
            parts = key_points[j]
            temp.append(parts['score'])
            position = parts['position']
            x, y = list(position.values())
            temp.append(x)
            temp.append(y)
        all_frames.append(temp)
    data_frame = pd.DataFrame(all_frames, columns=columns)
    return data_frame




#import json,requests
#url ="https://api.jsonbin.io/b/5e96def7435f5604bb41c7a5/1"
#
#r=requests.get(url)
#t=json.loads(r.content)
#
#json=convert_to_dataframe(t)
#
#test=single_test_features([json])
#
#


app = Flask(__name__)
@app.route('/')


@app.route('/',methods=['POST'])
def predict_api():
    json_data = request.json
#    if(json_data == '{}') :       
#        return jsonify({'Error' : 'No input'})
    print("data from json =",json_data)
    json = convert_to_dataframe(json_data)
    test=single_test_features([json])
    
    
    output = {'1': gestures[ann.predict(test)[0]], '2':gestures[knn.predict(test)[0]] ,
                '3': gestures[qda.predict(test)[0]] ,'4':gestures[rccv.predict(test)[0]]}

    return jsonify(output)

    
if __name__ == '__main__':
	app.run(debug=True)














