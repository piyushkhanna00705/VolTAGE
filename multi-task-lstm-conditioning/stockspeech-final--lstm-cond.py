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
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation , Masking, Bidirectional, TimeDistributed, Input,concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
import scipy.stats as stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error
import math
from tensorflow.keras.layers import dot, multiply, concatenate

with open('../cross-modal-attn/audio_featDict.pkl', 'rb') as f:
    audio_featDict=pickle.load(f)
    
with open('../cross-modal-attn/audio_featDictMark2.pkl', 'rb') as f:
    audio_featDictMark2=pickle.load(f)

with open('../cross-modal-attn/finbert_earnings.pkl', 'rb') as f:
    text_dict=pickle.load(f)
    
with open('./emb_3days_final.pkl', 'rb') as f:
    graph_embd_dict3=pickle.load(f)
with open('./emb_7days_final.pkl', 'rb') as f:
    graph_embd_dict7=pickle.load(f)
with open('./emb_15days_final.pkl', 'rb') as f:
    graph_embd_dict15=pickle.load(f)
with open('./emb_30days_final.pkl', 'rb') as f:
    graph_embd_dict30=pickle.load(f)
    
traindf= pd.read_csv("../cross-modal-attn/train_split_Avg_Series_WITH_LOG.csv")
testdf=pd.read_csv("../cross-modal-attn/test_split_Avg_Series_WITH_LOG.csv")
valdf=pd.read_csv("../cross-modal-attn/val_split_Avg_Series_WITH_LOG.csv")

single_traindf=pd.read_csv("../cross-modal-attn/train_split_SeriesSingleDayVol.csv")
single_testdf=pd.read_csv("../cross-modal-attn/test_split_SeriesSingleDayVol3.csv")
single_valdf=pd.read_csv("../cross-modal-attn/val_split_SeriesSingleDayVol3.csv")

#len(text_dict)
error_audio=[]
error_text=[]
error_graph=[]

def ModifyData(df,single_df):
    X_past=[]
    X_graph3=[]
    X_graph7=[]
    X_graph15=[]
    X_graph30=[]
    
    
    y_avg3days=[]
    y_avg7days=[]
    y_avg15days=[]
    y_avg30days=[]
    
    y_single3days=[]
    y_single7days=[]
    y_single15days=[]
    y_single30days=[]

    for index,row in df.iterrows():
        empty_graph_embd= np.zeros((200), dtype=np.float64)
        try:
            X_graph3.append(graph_embd_dict3[row['text_file_name'][:-9]])
            X_graph7.append(graph_embd_dict7[row['text_file_name'][:-9]])
            X_graph15.append(graph_embd_dict15[row['text_file_name'][:-9]])
            X_graph30.append(graph_embd_dict30[row['text_file_name'][:-9]])
        except KeyError:
            error_graph.append(row['text_file_name'][:-9])
            X_graph3.append(empty_graph_embd)
            X_graph7.append(empty_graph_embd)
            X_graph15.append(empty_graph_embd)
            X_graph30.append(empty_graph_embd)
        
        #Past 30 Days avg vol
        X_past.append(list(row[35:]))
        
        y_avg3days.append(float(row['future_3']))
        y_avg7days.append(float(row['future_7']))
        y_avg15days.append(float(row['future_15']))
        y_avg30days.append(float(row['future_30']))  
        
    for index,row in single_df.iterrows():
        y_single3days.append(float(row['future_Single_3']))
        y_single7days.append(float(row['future_Single_7']))
        y_single15days.append(float(row['future_Single_15']))
        y_single30days.append(float(row['future_Single_30']))
        
    X_past=np.array(X_past)
    X_graph3=np.nan_to_num(np.array(X_graph3))
    X_graph7=np.nan_to_num(np.array(X_graph7))
    X_graph15=np.nan_to_num(np.array(X_graph15))
    X_graph30=np.nan_to_num(np.array(X_graph30))
    
    
    
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
    
    X_past=np.nan_to_num(X_past)
    
    n_features = 1
    X_past = X_past.reshape((X_past.shape[0], X_past.shape[1], n_features))
        
    return X_past,X_graph3,X_graph7,X_graph15,X_graph30,y_avg3days,y_avg7days,y_avg15days,y_avg30days,y_single3days,y_single7days,y_single15days,y_single30days



X_train_past,X_train_graph3,X_train_graph7,X_train_graph15,X_train_graph30,y_train3days, y_train7days, y_train15days, y_train30days,y_train_single_3days, y_train_single_7days, y_train_single_15days, y_train_single_30days=ModifyData(traindf,single_traindf)

X_test_past,X_test_graph3,X_test_graph7,X_test_graph15,X_test_graph30, y_test3days, y_test7days, y_test15days, y_test30days,y_test_single_3days, y_test_single_7days, y_test_single_15days, y_test_single_30days=ModifyData(testdf,single_testdf)

X_val_past,X_val_graph3,X_val_graph7,X_val_graph15,X_val_graph30,y_val3days, y_val7days, y_val15days, y_val30days,y_val_single_3days, y_val_single_7days, y_val_single_15days, y_val_single_30days=ModifyData(valdf,single_valdf)

input_past_shape = (X_train_past.shape[1], X_train_past.shape[2])
input_graph_shape = (X_train_graph3.shape[1],)

#X_train_graph3.shape

duration_list=[]
batch_sizes=[]
epochs_list=[]
optimizer_list=[]
training_loss_list=[]
test_loss_list=[]
pearson_list=[]
spearman_list=[]

avg_day_mse_list=[]
single_day_mse_list=[]

def _get_tensor_shape(t):
    return t.shape

class ConditionalRNN(tf.keras.layers.Layer):

    # Arguments to the RNN like return_sequences, return_state...
    def __init__(self, units, cell=tf.keras.layers.LSTMCell, *args,
                 **kwargs):
        """
        Conditional RNN. Conditions time series on categorical data.
        :param units: int, The number of units in the RNN Cell
        :param cell: string, cell class or object (pre-instantiated). In the case of string, 'GRU',
        'LSTM' and 'RNN' are supported.
        :param args: Any parameters of the tf.keras.layers.RNN class, such as return_sequences,
        return_state, stateful, unroll...
        """
        super().__init__()
        self.units = units
        self.final_states = None
        self.init_state = None
        if isinstance(cell, str):
            if cell.upper() == 'GRU':
                cell = tf.keras.layers.GRUCell
            elif cell.upper() == 'LSTM':
                cell = tf.keras.layers.LSTMCell
            elif cell.upper() == 'RNN':
                cell = tf.keras.layers.SimpleRNNCell
            else:
                raise Exception('Only GRU, LSTM and RNN are supported as cells.')
        self._cell = cell if hasattr(cell, 'units') else cell(units=units)
        self.rnn = tf.keras.layers.RNN(cell=self._cell, *args, **kwargs)

        # single cond
        self.cond_to_init_state_dense_1 = tf.keras.layers.Dense(units=self.units)

        # multi cond
        max_num_conditions = 10
        self.multi_cond_to_init_state_dense = []
        for i in range(max_num_conditions):
            self.multi_cond_to_init_state_dense.append(tf.keras.layers.Dense(units=self.units))
        self.multi_cond_p = tf.keras.layers.Dense(1, activation=None, use_bias=True)

    def _standardize_condition(self, initial_cond):
        initial_cond_shape = initial_cond.shape
        if len(initial_cond_shape) == 2:
            initial_cond = tf.expand_dims(initial_cond, axis=0)
        first_cond_dim = initial_cond.shape[0]
        if isinstance(self._cell, tf.keras.layers.LSTMCell):
            if first_cond_dim == 1:
                initial_cond = tf.tile(initial_cond, [2, 1, 1])
            elif first_cond_dim != 2:
                raise Exception('Initial cond should have shape: [2, batch_size, hidden_size]\n'
                                'or [batch_size, hidden_size]. Shapes do not match.', initial_cond_shape)
        elif isinstance(self._cell, tf.keras.layers.GRUCell) or isinstance(self._cell, tf.keras.layers.SimpleRNNCell):
            if first_cond_dim != 1:
                raise Exception('Initial cond should have shape: [1, batch_size, hidden_size]\n'
                                'or [batch_size, hidden_size]. Shapes do not match.', initial_cond_shape)
        else:
            raise Exception('Only GRU, LSTM and RNN are supported as cells.')
        return initial_cond

    def __call__(self, inputs, *args, **kwargs):
        """
        :param inputs: List of n elements:
                    - [0] 3-D Tensor with shape [batch_size, time_steps, input_dim]. The inputs.
                    - [1:] list of tensors with shape [batch_size, cond_dim]. The conditions.
        In the case of a list, the tensors can have a different cond_dim.
        :return: outputs, states or outputs (if return_state=False)
        """
        assert isinstance(inputs, list) and len(inputs) >= 2
        x = inputs[0]
        cond = inputs[1:]
        if len(cond) > 1:  # multiple conditions.
            init_state_list = []
            for ii, c in enumerate(cond):
                init_state_list.append(self.multi_cond_to_init_state_dense[ii](self._standardize_condition(c)))
            multi_cond_state = self.multi_cond_p(tf.stack(init_state_list, axis=-1))
            multi_cond_state = tf.squeeze(multi_cond_state, axis=-1)
            self.init_state = tf.unstack(multi_cond_state, axis=0)
        else:
            cond = self._standardize_condition(cond[0])
            if cond is not None:
                self.init_state = self.cond_to_init_state_dense_1(cond)
                self.init_state = tf.unstack(self.init_state, axis=0)
        out = self.rnn(x, initial_state=self.init_state, *args, **kwargs)
        if self.rnn.return_state:
            outputs, h, c = out
            final_states = tf.stack([h, c])
            return outputs, final_states
        else:
            return out

class LSTM_conditioned(tf.keras.Model):
        def __init__(self):
            super(LSTM_conditioned, self).__init__()
            self.cond = ConditionalRNN(units=200, cell='LSTM', dtype=tf.float32)
#             self.out1 = tf.keras.layers.Dense(units=29, activation='linear')
#             self.out2 = tf.keras.layers.Dense(units=29, activation='linear')

        def call(self, inputs, **kwargs):
            o = self.cond(inputs)
#             o = self.out1(o)
#             o2 = self.out1(o)
            return o

def train(duration,mu,t_bilstm1,fc1,a_bilstm1, c_bilstm, c_fc ,y_train, y_val, y_test,y_train_single, y_val_single, y_test_single, batch_size, epochs, learning_rate):
    
    a_fc1=fc1
    t_fc1=fc1

    input_gcn= Input(shape=input_graph_shape)
    input_past = Input(shape=input_past_shape)
    
    if duration==3:
        X_train_graph=X_train_graph3
        X_val_graph=X_val_graph3
        X_test_graph=X_test_graph3
    elif duration==7:
        X_train_graph=X_train_graph7
        X_val_graph=X_val_graph7
        X_test_graph=X_test_graph7
    elif duration==15:
        X_train_graph=X_train_graph15
        X_val_graph=X_val_graph15
        X_test_graph=X_test_graph15
    else:
        X_train_graph=X_train_graph30
        X_val_graph=X_val_graph30
        X_test_graph=X_test_graph30
    
    
    lstm_cond=LSTM_conditioned()
    lstm_cond_out=lstm_cond.call([input_past, input_gcn])
    
    C_final1 = Dense(units=1, activation='linear',name='avg_vol_op')(lstm_cond_out)
    C_final2 = Dense(units=1, activation='linear',name='single_vol_op')(lstm_cond_out)
    
    
    model = Model(inputs=[input_gcn,input_past], outputs=[C_final1,C_final2])
    
    adam = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile( optimizer=adam,loss={'avg_vol_op': 'mean_squared_error', 'single_vol_op': 'mean_squared_error'},
              loss_weights={'avg_vol_op': mu, 'single_vol_op': (1-mu)})
    
#     print(model.summary())
    
    history=model.fit(
                      [X_train_graph,X_train_past],
                      [y_train,y_train_single],batch_size=batch_size,
                      epochs=epochs,
                      validation_data=([X_val_graph,X_val_past], [y_val,y_val_single])
                     )
    
   
    test_loss = model.evaluate([X_test_graph,X_test_past],[y_test,y_test_single],batch_size=batch_size)
    train_loss = model.evaluate([X_train_graph,X_train_past],[y_train,y_train_single],batch_size=batch_size)
    
    print()
    print("Train loss  : {train_loss}".format(train_loss = train_loss))
    print("Test loss : {test_loss}".format(test_loss = test_loss))
    
    print()
    y_pred_avg,y_pred_single = model.predict([X_test_graph,X_test_past])
    
    avg_day_mse=mean_squared_error(y_test, y_pred_avg)
    single_day_mse=mean_squared_error(y_test_single, y_pred_single)
    
    avg_day_mse_list.append(avg_day_mse)
    single_day_mse_list.append(single_day_mse)
    
    pickle.dump(avg_day_mse_list,open('./GCN_LSTM_cond_avg_day_mse_list.pkl','wb'))
    pickle.dump(single_day_mse_list,open('./GCN_LSTM_cond_single_day_mse_list.pkl','wb'))
    
    print("Duration="+str(duration))
    print("MSE-Avg-Day="+str(avg_day_mse))
    print("MSE-Single-Day="+str(single_day_mse))
    duration_list.append('Future_'+str(duration))
    batch_sizes.append(batch_size)
    epochs_list.append(epochs)
    optimizer_list.append('adam_'+str(learning_rate))
    training_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
#     pearson_list.append(PearsonCorrel)
#     spearman_list.append(Spearmancoef)

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
    print("i="+str(i))
    train(duration=3,mu=0.7,t_bilstm1=100,fc1=100,a_bilstm1=100,c_bilstm=100,c_fc=100 ,y_train=y_train3days, y_val=y_val3days, y_test=y_test3days,y_train_single=y_train_single_3days, y_val_single=y_val_single_3days, y_test_single=y_test_single_3days, batch_size=32, epochs=50, learning_rate=0.001)

avg_day_mse_df=pd.DataFrame(avg_day_mse_list)
avg_day_mse_df.to_csv('./3GCN_LSTM_boxplot_cond_avg_day_mse_df.csv')

single_day_mse_df=pd.DataFrame(single_day_mse_list)
single_day_mse_df.to_csv('./3GCN_LSTM_boxplot_cond_single_day_mse_df.csv')

