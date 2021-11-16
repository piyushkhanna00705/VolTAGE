
import numpy as np 
import pandas as pd
import pickle
import os
import datetime
from tqdm import tqdm
from statistics import mean 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation , Masking, Bidirectional, TimeDistributed, Input,concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import scipy.stats as stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from keras.models import Model
import math
from keras.layers import dot, multiply, concatenate

with open('./audio_featDict.pkl', 'rb') as f:
    audio_featDict=pickle.load(f)
    
with open('./audio_featDictMark2.pkl', 'rb') as f:
    audio_featDictMark2=pickle.load(f)

with open('./finbert_earnings.pkl', 'rb') as f:
    text_dict=pickle.load(f)
    
traindf= pd.read_csv("./train_split_Avg_Series_WITH_LOG.csv")
testdf=pd.read_csv("./test_split_Avg_Series_WITH_LOG.csv")
valdf=pd.read_csv("./val_split_Avg_Series_WITH_LOG.csv")

single_traindf= pd.read_csv("./train_split_SeriesSingleDayVol3.csv")
single_testdf=pd.read_csv("./test_split_SeriesSingleDayVol3.csv")
single_valdf=pd.read_csv("./val_split_SeriesSingleDayVol3.csv")

avg_day_mse_list=[]
single_day_mse_list=[]

error_audio=[]
error_text=[]

def ModifyData(df,single_df):
    X=[]
    X_text=[]
    
    y_avg3days=[]
    y_avg7days=[]
    y_avg15days=[]
    y_avg30days=[]
    
    y_single3days=[]
    y_single7days=[]
    y_single15days=[]
    y_single30days=[]

    for index,row in df.iterrows():
        try:
            X_text.append(text_dict[row['text_file_name']])
        except KeyError:
            error_text.append(row['text_file_name'])

        lstm_matrix_temp = np.zeros((520, 26), dtype=np.float64)
        i=0
        
        try:
            speaker_list=list(audio_featDict[row['text_file_name']])
            speaker_list=sorted(speaker_list, key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2])))
            for sent in speaker_list:
                lstm_matrix_temp[i, :]=audio_featDict[row['text_file_name']][sent]+audio_featDictMark2[row['text_file_name']][sent]
                i+=1
            X.append(lstm_matrix_temp)

        except KeyError:
            Padded=np.zeros((520, 26), dtype=np.float64)
            X.append(Padded)
            error_audio.append(row['text_file_name'][:-9])
            

        y_avg3days.append(float(row['future_3']))
        y_avg7days.append(float(row['future_7']))
        y_avg15days.append(float(row['future_15']))
        y_avg30days.append(float(row['future_30']))  
        
    for index,row in single_df.iterrows():
        y_single3days.append(float(row['future_Single_3']))
        y_single7days.append(float(row['future_Single_7']))
        y_single15days.append(float(row['future_Single_15']))
        y_single30days.append(float(row['future_Single_30']))
        
    X=np.array(X)
    X_text=np.array(X_text)
    
    y_avg3days=np.array(y_avg3days)
    y_avg7days=np.array(y_avg7days)
    y_avg15days=np.array(y_avg15days)
    y_avg30days=np.array(y_avg30days)
    
    y_single3days=np.array(y_single3days)
    y_single7days=np.array(y_single7days)
    y_single15days=np.array(y_single15days)
    y_single30days=np.array(y_single30days)
    
#     print(np.sum(y_3days))
#     print(np.sum(y_7days))
#     print(np.sum(y_15days))
#     print(np.sum(y_30days))
    
    X=np.nan_to_num(X)
    X_text=np.nan_to_num(X_text)
        
    return X,X_text,y_avg3days,y_avg7days,y_avg15days,y_avg30days,y_single3days,y_single7days,y_single15days,y_single30days



X_train_audio,X_train_text,y_train3days, y_train7days, y_train15days, y_train30days,y_train_single_3days, y_train_single_7days, y_train_single_15days, y_train_single_30days=ModifyData(traindf,single_traindf)

X_test_audio,X_test_text, y_test3days, y_test7days, y_test15days, y_test30days,y_test_single_3days, y_test_single_7days, y_test_single_15days, y_test_single_30days=ModifyData(testdf,single_testdf)

X_val_audio,X_val_text,y_val3days, y_val7days, y_val15days, y_val30days,y_val_single_3days, y_val_single_7days, y_val_single_15days, y_val_single_30days=ModifyData(valdf,single_valdf)

input_audio_shape = (X_train_audio.shape[1], X_train_audio.shape[2])
input_text_shape = (X_train_text.shape[1],X_train_text.shape[2])
#np.argwhere(np.isnan(y_train3days))
#X_train_audio.shape
#X_train_text.shape

def bi_modal_attention(x, y):
    
    ''' 
    .  stands for dot product 
    *  stands for elemwise multiplication
    {} stands for concatenation
        
    m1 = x . transpose(y) ||  m2 = y . transpose(x) 
    n1 = softmax(m1)      ||  n2 = softmax(m2)
    o1 = n1 . y           ||  o2 = m2 . x
    a1 = o1 * x           ||  a2 = o2 * y
       
    return {a1, a2}
        
    '''
     
    m1 = dot([x, y], axes=[2, 2])
    n1 = Activation('softmax')(m1)
    o1 = dot([n1, y], axes=[2, 1])
    a1 = multiply([o1, x])

    m2 = dot([y, x], axes=[2, 2])
    n2 = Activation('softmax')(m2)
    o2 = dot([n2, x], axes=[2, 1])
    a2 = multiply([o2, y])

    return concatenate([a1, a2])

def self_attention(x):
    
    ''' 
    .  stands for dot product 
    *  stands for elemwise multiplication
        
    m = x . transpose(x)
    n = softmax(m)
    o = n . x  
    a = o * x           
       
    return a
        
    '''

    m = dot([x, x], axes=[2,2])
    n = Activation('softmax')(m)
    o = dot([n, x], axes=[2,1])
    a = multiply([o, x])
        
    return a

duration_list=[]
batch_sizes=[]
epochs_list=[]
optimizer_list=[]
training_loss_list=[]
test_loss_list=[]
pearson_list=[]
spearman_list=[]

def train(duration,t_bilstm1,fc1,a_bilstm1, c_bilstm, c_fc ,y_train, y_val, y_test,y_train_single, y_val_single, y_test_single, batch_size, epochs, learning_rate):
    
    a_fc1=fc1
    t_fc1=fc1
    
    input_text = Input(shape=input_text_shape) 
    input_audio = Input(shape=input_audio_shape)
    mask_text=Masking(mask_value=0)(input_text)
    T_bilstm1=Bidirectional(LSTM(units=t_bilstm1, dropout=0.8, recurrent_dropout=0.8, activation='tanh' ,return_sequences=True))(mask_text)
    T_drop1=Dropout(0.8)(T_bilstm1)
    T_fc1=TimeDistributed(Dense(units=t_fc1,activation='relu'))(T_drop1)
    
    
    mask_audio=Masking(mask_value=0)(input_audio)
    A_bilstm1=Bidirectional(LSTM(units=a_bilstm1, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))(mask_audio)
    A_drop1=Dropout(0.5)(A_bilstm1)
    A_fc1=TimeDistributed(Dense(units=a_fc1,activation='relu'))(A_drop1)
    
    
    combined_at = bi_modal_attention(T_fc1,A_fc1)
    
    C_at_bilstm=Bidirectional(LSTM(units=c_bilstm, dropout=0, recurrent_dropout=0, activation='tanh'))(combined_at)
    
    C_fc=Dense(units=c_fc,activation='tanh')(C_at_bilstm)
    
    C_final1=Dense(units=1, activation='linear',name='avg_vol_op')(C_fc)
    
    C_final2=Dense(units=1, activation='linear',name='single_vol_op')(C_fc)
    
    
    model = Model(inputs=[input_text,input_audio], outputs=[C_final1,C_final2])
    
    adam = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile( optimizer=adam,loss={'avg_vol_op': 'mean_squared_error', 'single_vol_op': 'mean_squared_error'},
              loss_weights={'avg_vol_op': 0.5, 'single_vol_op': 0.5})
    
#     print(model.summary())
    
    history=model.fit(
                      [X_train_text,X_train_audio],
                      [y_train,y_train_single],batch_size=batch_size,
                      epochs=epochs,
                      validation_data=([X_val_text,X_val_audio], [y_val,y_val_single])
                     )
    
   
    test_loss = model.evaluate([X_test_text,X_test_audio],[y_test,y_test_single],batch_size=batch_size)
    train_loss = model.evaluate([X_train_text,X_train_audio],[y_train,y_train_single],batch_size=batch_size)
    
    print()
    print("Train loss  : {train_loss}".format(train_loss = train_loss))
    print("Test loss : {test_loss}".format(test_loss = test_loss))
    
    print()
    y_pred_avg,y_pred_single = model.predict([X_test_text,X_test_audio])
    
    
    avg_day_mse=mean_squared_error(y_test, y_pred_avg)
    single_day_mse=mean_squared_error(y_test_single, y_pred_single)
    
    avg_day_mse_list.append(avg_day_mse)
    single_day_mse_list.append(single_day_mse)
    
    pickle.dump(avg_day_mse_list,open('./AT_attn3_avg_day_mse_list.pkl','wb'))
    pickle.dump(single_day_mse_list,open('./AT_attn3_single_day_mse_list.pkl','wb'))
    
    print("Duration="+str(duration))
    print("MSE-Avg-Day="+str(avg_day_mse))
    print("MSE-Single-Day="+str(single_day_mse))
    
    duration_list.append('Future_'+str(duration))
    batch_sizes.append(batch_size)
    epochs_list.append(epochs)
    optimizer_list.append('adam_'+str(learning_rate))
    training_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    save_path = "duration="+str(duration)+"epochs="+str(epochs)+"_learning-rate"+str(learning_rate)
    save_pkl_avg="y_pred_avg_"+save_path+".pkl"
    
    with open(save_pkl_avg,'wb') as f:
        pickle.dump(y_pred_avg,f)
    
    save_pkl_single="y_pred_single_"+save_path+".pkl"
    with open(save_pkl_single,'wb') as f:
        pickle.dump(y_pred_single,f)
        
    model.save(save_path+"_model.h5")
    plt.savefig(save_path+".png")
    plt.show()
    plt.close()
    return

for i in range(10):
    print(i+1)
    train(duration=3,t_bilstm1=100,fc1=100,a_bilstm1=100,c_bilstm=100,c_fc=50 ,y_train=y_train3days, y_val=y_val3days, y_test=y_test3days,y_train_single=y_train_single_3days, y_val_single=y_val_single_3days, y_test_single=y_test_single_3days, batch_size=32, epochs=50, learning_rate=0.001)

avg_day_mse_df=pd.Dataframe(avg_day_mse_list)
avg_day_mse_df.to_csv('./3avg_day_mse_df.csv')

single_day_mse_df=pd.Dataframe(single_day_mse_list)
single_day_mse_df.to_csv('./3single_day_mse_df.csv')

