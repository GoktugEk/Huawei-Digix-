# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn import preprocessing 
from tensorflow import metrics
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import time

def get_model():
    model = Sequential([
        Dense(units=8,input_shape=(19,), activation='relu'),
        Dense(units=16,activation='relu'),
        Dense(units=32,activation='relu'),
        Dense(units=16,activation='relu'),
        Dense(units=8,activation='relu'),
        Dense(units=4,activation='relu'),
        Dense(units=2,activation='relu'),
        Dense(units=1,activation='sigmoid')
    ]
    )
    
    model.compile(loss='binary_crossentropy', optimizer= 'Adam',metrics=[ tf.keras.metrics.AUC()])
    
    return model

def train():

    
    scalar = preprocessing.StandardScaler()  


    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='auc', 
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)

    for i in range(8):
        df = pd.read_csv(
            "six_million_" + str(i) + ".csv",
            sep='|',
            header = 0,
            error_bad_lines=False,
            )
        
        
        labels = df.iloc[:,[1]]
        labels = np.array(labels) 
    
        
        features = df.iloc[:, 2:]
        features = np.array(features)


        scaled_feats = scalar.fit_transform(features)
    
        features = None
        
        model.fit(scaled_feats,labels,callbacks=[early_stopping],shuffle=True,validation_split=0.15,batch_size=16, epochs=30)

        line += row

   model.save("model_mk4")
    return scalar



def test(scalar):
    t1 = time.time()
    test_df = pd.read_csv(
        "test_data_B.csv",
        sep='|',
        usecols=[11,12,13,14,15,17,18,19,20,21,22,23,24,26,27,31,33,34,35],
        header = 0,
        names=['row', 'uid', 'task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id', 'inter_typ_cd', 'slot_id', 'spread_app_id', 'tags', 'app_first_class', 'app_second_class', 'age', 'city', 'city_rank', 'device_name', 'device_size', 'career', 'gender', 'net_type', 'residence', 'his_app_size', 'his_on_shelf_time', 'app_score', 'emui_dev', 'list_time', 'device_price', 'up_life_duration', 'up_membership_grade', 'membership_life_duration', 'consume_purchase', 'communication_onlinerate', 'communication_avgonline_30d', 'indu_name', 'pt_d']
        )
    

    scaled = scalar.fit_transform(test_df)
    
    predicted = model.predict(scaled)
    t2 = time.time()
    print(t2-t1)
    print(predicted.shape,predicted)
    
    f = open("submission.csv","w+")
    
    for i,pred in enumerate(predicted):
        f.write(str(i+1)+","+str(pred[0])+"\n")
    
    f.close()


def preprocess_the_data():
    line = 0
    row = 6000000
    i = 0
    filename = "six_million_"  
    undersample = RandomUnderSampler(0.5)
    oversample  = SMOTE(sampling_strategy=0.35)

    while row:
        try:
            df = pd.read_csv(
                "train_data.csv",
                skiprows =line,
                nrows= row,
                sep='|',
                usecols=[0,11,12,13,14,15,17,18,19,20,21,22,23,24,26,27,31,33,34,35],
                header = 0,
                error_bad_lines=False,
                names=['label', 'uid', 'task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id', 'inter_typ_cd', 'slot_id', 'spread_app_id', 'tags', 'app_first_class', 'app_second_class', 'age', 'city', 'city_rank', 'device_name', 'device_size', 'career', 'gender', 'net_type', 'residence', 'his_app_size', 'his_on_shelf_time', 'app_score', 'emui_dev', 'list_time', 'device_price', 'up_life_duration', 'up_membership_grade', 'membership_life_duration', 'consume_purchase', 'communication_onlinerate', 'communication_avgonline_30d', 'indu_name', 'pt_d']
                )
        except:
            df = pd.read_csv(
                "train_data.csv",
                skiprows =line,
                sep='|',
                usecols=[0,11,12,13,14,15,17,18,19,20,21,22,23,24,26,27,31,33,34,35],
                header = 0,
                error_bad_lines=False,
                names=['label', 'uid', 'task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id', 'inter_typ_cd', 'slot_id', 'spread_app_id', 'tags', 'app_first_class', 'app_second_class', 'age', 'city', 'city_rank', 'device_name', 'device_size', 'career', 'gender', 'net_type', 'residence', 'his_app_size', 'his_on_shelf_time', 'app_score', 'emui_dev', 'list_time', 'device_price', 'up_life_duration', 'up_membership_grade', 'membership_life_duration', 'consume_purchase', 'communication_onlinerate', 'communication_avgonline_30d', 'indu_name', 'pt_d']
                )
            row = 0
        

        labels = df.iloc[:,[0]]
        labels = np.array(labels) 
  
        
        features = df.iloc[:, 1:]
        features = np.array(features)
        

        features,labels = oversample.fit_resample(features,labels)

        features,labels = undersample.fit_resample(features,labels)

        labels = labels.reshape(-1,1)
        df = np.concatenate((labels,features),axis=1)
        
        df = pd.DataFrame(df)

        df.to_csv(filename+str(i)+".csv",sep="|")
        
        i+=1



model = get_model()
scalar = train()
test(scalar)




