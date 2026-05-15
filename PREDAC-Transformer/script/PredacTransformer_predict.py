import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution( )
import numpy as np
import pandas as pd
import os
import glob
import argparse
from sklearn.utils import class_weight
import time
from sklearn.metrics import matthews_corrcoef,f1_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from sklearn import model_selection
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc, roc_auc_score
import collections
import matplotlib.pyplot as plt
import seaborn as sns
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def early_stopping(info,monitor,patience):
    df = info
    df['epoch']=df.index+1
    max_PRC_dev = 0.0  
    last_improve = -1  

    for epoch, PRC_dev in df[['epoch', monitor]].itertuples(index=False):
        if PRC_dev > max_PRC_dev:
            last_improve = epoch
            max_PRC_dev = PRC_dev 
        if epoch - last_improve >= patience:
            break
    return last_improve


def plot_attention(filename, dflabel, dfpredict, label, outdir_0, fold):
    """
    Plot attention weights and save the results to a CSV file.

    """
    # Create a DataFrame to store real and predicted labels
    data = pd.DataFrame()
    data['real'] = list(dflabel)
    data['predict'] = list(dfpredict)
    data['label'] = 0  # Initialize label column
    data['label'][data['real'] == data['predict']] = 1  # Mark correct predictions
    # Print the count of correct and incorrect predictions
    print(data['label'].value_counts())

    # Select data based on the label type
    if label == 'all':
        length_index = np.array(data.index)
    elif label == '1':
        length_index = np.array(data[data.label == 1].index)
    elif label == '0':
        length_index = np.array(data[data.label == 0].index)
    else:
        raise ValueError("Invalid label type. Use 'all', '1', or '0'.")
     # Filter attention weights based on selected indices
    testa = filename
    print("Original attention weights shape:", testa.shape)
    testa = testa[length_index, :, :, :]
    print("Filtered attention weights shape:", testa.shape)
    
    # Sum attention weights across specified axes
    testn = testa.sum(axis=(0, 1, 2))  # Sum across all axes except the last one
    print("Summed attention weights shape:", testn.shape)

    # Prepare data for saving to CSV
    N = len(testn)
    index = range(1, N + 1)
    df = pd.DataFrame(testn)
    df['position'] = index
    # Save the DataFrame to a CSV file
    output_filename = f"{outdir_0}/attention_score_{label}.csv" #f"{outdir_0}/{fold}_attention_score_{label}.csv"
    df.to_csv(output_filename, index=False)
    print(f"Saved attention weights to {output_filename}") 
      
class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, rate=0.5):
    super(Encoder, self).__init__()
    self.d_model = d_model
    self.num_layers = num_layers    
    self.pos_encoding = positional_encoding(1000, self.d_model) 
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
        
  def call(self, x, training=True, mask=None):
    seq_len = tf.shape(x)[1]
    x += self.pos_encoding[:, :seq_len, :]
    x = self.dropout(x, training=training)
    enc_self_attns=[]
    for i in range(self.num_layers):
      x,attn_weights = self.enc_layers[i](x, training, mask)
      enc_self_attns.append(attn_weights)
    return x ,enc_self_attns



def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],np.arange(d_model)[np.newaxis, :],d_model)
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
  pos_encoding = angle_rads[np.newaxis, ...]
  return tf.cast(pos_encoding, dtype=tf.float32)

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.5):
    super(EncoderLayer, self).__init__()
    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6) 
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, training=True, mask=None):
    attn_output, attn_weights = self.mha(x, x, x, mask)  
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  
    ffn_output = self.ffn(out1) 
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  
    return out2,attn_weights

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  
      tf.keras.layers.Dense(d_model)  
  ])


#multi-head attention
class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    assert d_model % self.num_heads == 0
    self.depth = d_model // self.num_heads
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    self.dense = tf.keras.layers.Dense(d_model)
        
  def split_heads(self, x, batch_size):
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask=None):
    batch_size = tf.shape(q)[0]
    q = self.wq(q)  
    k = self.wk(k)  
    v = self.wv(v) 
    
    q = self.split_heads(q, batch_size) 
    k = self.split_heads(k, batch_size)  
    v = self.split_heads(v, batch_size)  
    
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  
    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  
    output = self.dense(concat_attention)  
    return output, attention_weights

def scaled_dot_product_attention(q, k, v, mask=None):
  matmul_qk = tf.matmul(q, k, transpose_b=True)  
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  
  output = tf.matmul(attention_weights, v)  
  return output, attention_weights


def binary_focal_loss(gamma=2, alpha=0.25):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)
    def binary_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true*alpha + (tf.ones_like(y_true)-y_true)*(1-alpha)
        p_t = y_true*y_pred + (tf.ones_like(y_true)-y_true)*(tf.ones_like(y_true)-y_pred) + tf.keras.backend.epsilon()
        focal_loss = - alpha_t * tf.pow((tf.ones_like(y_true)-p_t),gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)
    return binary_focal_loss_fixed

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_confusion_matrix(y_true, y_pred):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(y_pred.shape[0]):
        if y_pred[i,1] >= 0.5:
            y_pred[i,1]=1
            y_pred[i,0]=0
        else:
            y_pred[i,1]=0
            y_pred[i,0]=1

    TP=sum(y_true[:,1] * y_pred[:,1])
    TN=sum(y_true[:,0] * y_pred[:,0])
    FP=sum(y_true[:,0] * y_pred[:,1])
    FN=sum(y_true[:,1] * y_pred[:,0])

    return TP, FP, TN, FN


def get_PredacTransformer_model(shape_0,shape_1,encoder_layer,encoder_head):
    inputESM=tf.keras.layers.Input(shape=(shape_0,shape_1)) 

    sequence=tf.keras.layers.Dense(32)(inputESM)
    sequence=tf.keras.layers.Dense(16)(sequence)
    sequence,enc_self_attns = Encoder(encoder_layer, 16, encoder_head, 64, rate=0.1)(sequence)
    sequence=tf.keras.layers.Flatten(input_shape=(shape_0,64))(sequence)
    feature=tf.keras.layers.Dense(512,activation='relu')(sequence)
    feature=tf.keras.layers.Dense(256,activation='relu')(feature)
    feature=tf.keras.layers.Dense(128,activation='relu')(feature)
    feature=tf.keras.layers.Dropout(0.1)(feature)
    y=tf.keras.layers.Dense(2, activation='softmax')(feature)
    PredacTransformer_model=tf.keras.models.Model(inputs=inputESM,outputs=[y,enc_self_attns])
    adam=tf.keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08,clipnorm=1.0,clipvalue=0.5,decay=1e-8)
    PredacTransformer_model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy',f1_m])
    PredacTransformer_model.summary()
    return PredacTransformer_model


if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--indicator',default='training', metavar='training or fine_tune')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--model_dir', default=None)
    parser.add_argument('--save_dir', default='results')
    parser.add_argument('--shape_0',default=329,type=int)
    parser.add_argument('--shape_1',default=654,type=int)
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--csv_file', required=True)
    parser.add_argument('--filename', required=True)
    parser.add_argument('--epoch', default=200,required=False,type=int)
    parser.add_argument('--outdir',default=None)
    parser.add_argument('--fold',type=int,default=1)
    parser.add_argument('--encoder_layer',default=2,type=int)
    parser.add_argument('--encoder_head',default=4,type=int)
    parser.add_argument('--number_columns', default=654,type=int)
    parser.add_argument('--mode',default='fulllength')
    parser.add_argument('--subtype', required=True)
    args = parser.parse_args()
    print(args)

    # load dataset

    filename=args.filename
    encoder_layer=args.encoder_layer
    encoder_head=args.encoder_head
    number_columns=args.number_columns
    shape_1=args.shape_1


    model_path=args.model_dir
    outdir_0=args.outdir
    fold=args.fold
    mode=args.mode
    if args.subtype =='H1N1':
        shape_0=327
        encoder_head=4
    elif args.subtype =='H3N2':
        shape_0=329
        encoder_head=8
    if args.mode != 'fulllength':
        print(test_data.shape)
        test_data = test_data[:, important_positions, :]
        print(test_data.shape)

    seed=100 
    test_data=np.load(args.input_dir+'/'+filename)
    print(test_data.shape)


    test_x=test_data[:,:,:number_columns]
    test_y_0=test_data[:,0,number_columns]

    test_y=np.zeros((test_x.shape[0],2))


    for i in range(test_x.shape[0]):
        test_y[i,0]=1-test_y_0[i]
        test_y[i,1]=test_y_0[i]



    PredacTransformer_model=get_PredacTransformer_model(shape_0,shape_1,encoder_layer,encoder_head)
    PredacTransformer_model.load_weights(model_path)
    print(PredacTransformer_model.predict(test_x))

    pred_y,*enc_self_attns_list=PredacTransformer_model.predict(test_x)
    #print(fold)
    print(pred_y[:,0])
    print(pred_y[:,1])

    

    pred_y_0=pred_y[:,1]
    print(pred_y_0)  
    pred_y_0_np=np.array(pred_y_0)
    true_y_np=np.array(test_y_0)
    all_np=np.array(list(zip(true_y_np,pred_y_0_np)))
    #np.save(outdir_0+'/'+str(fold)+'_pred.npy',all_np)

    pred_y_0_np_01=np.where(pred_y_0_np >=0.5,1,0)
    seq_all=pd.read_csv(args.input_dir+'/'+args.csv_file,sep='\t')
    seq_all['predict']=pred_y_0_np_01
    seq_all['prob']=pred_y_0_np
    seq_all.to_csv(outdir_0+'/pred.csv')


    Y_Pred_new= pred_y_0_np_01
    idx_layer=1
    for enc_self_attns in enc_self_attns_list:  
        print(enc_self_attns.shape)
        if not os.path.exists(outdir_0+'/layer'+str(idx_layer)):
            os.makedirs(outdir_0+'/layer'+str(idx_layer))
        #np.save(outdir_0+'/layer'+str(idx_layer)+'/'+str(fold)+'_attention_weight.npy', enc_self_attns)
        
        print(test_y_0)
        print(Y_Pred_new)
        plot_attention(enc_self_attns,test_y_0,Y_Pred_new,'1',outdir_0+'/layer'+str(idx_layer),fold)
        #plot_attention(enc_self_attns,test_y_0,Y_Pred_new,'0',outdir_0+'/layer'+str(idx_layer),fold)  
        #plot_attention(enc_self_attns,test_y_0,Y_Pred_new,'all',outdir_0+'/layer'+str(idx_layer),fold)    
        idx_layer=idx_layer+1
    
    TP, FP, TN, FN=get_confusion_matrix(test_y, pred_y)

    print(get_confusion_matrix(test_y, pred_y))
    ps_dict={}
    ps_dict={'TP':TP,'FP':FP,'TN':TN,'FN':FN}
    ps=pd.DataFrame.from_dict(ps_dict,orient='index').T
    #ps['fold']=fold

    ps['AUC']=roc_auc_score(all_np[:,0], all_np[:,1])
    ps.to_csv(outdir_0+'/'+'log.csv',mode='a+',index=False)
    

    # os._exit(0)
    # sys.exit(0)
 
 
