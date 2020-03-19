## Functions
# get_model - create NeuMF model with inputs layers, reg_layers, reg_mf
# maskAware
# Seperate embeddings for user and item
# leaky Relu 

import numpy as np
import tensorflow as tf
from numpy.random import seed
from tensorflow import set_random_seed
seed(1)
set_random_seed(1)

from keras import initializers
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Embedding, Input, Dense, merge, Flatten, concatenate, multiply, dot, Reshape, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
import keras.callbacks
import keras
from keras import backend as K


months = 13
supplier = 563
wgs = 230
mkt = 9

months_emb = round(months ** 0.25)
supplier_emb = round(supplier ** 0.25)
wgs_emb = round(wgs ** 0.25)
mkt_emb = round(mkt ** 0.25)

max_length = 200 # how many user_clicks are included in history - for padding

## Mask aware mean in averaging the user history ! 
## Seperat Embeddings for different uses !

def mask_aware_mean(x):
# recreate the masks - all zero rows have been masked
    mask = K.not_equal(K.sum(K.abs(x), axis=2, keepdims=True), 0)
    # number of that rows are not all zeros
    n = K.sum(K.cast(mask, 'float32'), axis=1, keepdims=False)
    # compute mask-aware mean of x
    x_mean = K.sum(x, axis=1, keepdims=False) / n
    return x_mean 

lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)


def get_model(layers=[50], reg_layers=[0], reg_mf=0):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    ### Input variables
    # Context
    month = Input(shape = (1,), dtype = "float32", name = "month")
    day_online = Input(shape = (1,), dtype = "float32", name = "days_online")
    # Item
    item_anbieter = Input(shape = (1,), dtype = "float32", name = "item_anbieter")
    item_mkt = Input(shape = (1,), dtype = "float32", name = "item_mkt")
    item_wg = Input(shape = (1,), dtype = "float32", name = "item_wg")
    item_preis = Input(shape = (1,), dtype = "float32", name = "item_preis")
    item_ve = Input(shape = (1,), dtype = "float32", name = "item_ve")
    item_text = Input(shape = (150,), dtype = "float32", name = "item_text")

    # User
    user_mkt = Input(shape = (1,), dtype = "float32", name = "user_mkt")
    user_anbieter = Input(shape = (max_length,), dtype = "float32", name = "user_anbieter")
    user_anbietermkt = Input(shape = (max_length,), dtype = "float32", name = "user_anbietermkt")
    user_wg = Input(shape = (max_length,), dtype = "float32", name = "user_wg")
    user_preis = Input(shape = (1,), dtype = "float32", name = "user_preis")
    user_ve = Input(shape = (1,), dtype = "float32", name = "user_ve")
    user_text = Input(shape = (150,), dtype = "float32", name = "user_text")



    ### Embedding layer
    # MF
    # Context
    MF_Embedding_Context_month = Embedding(input_dim=months, output_dim=months_emb, name='mf_embedding_month',
                                  embeddings_initializer=initializers.random_normal(),
                                  embeddings_regularizer=l2(reg_mf), input_length=1)
    
    # Item
    
    MF_Embedding_Item_anbieter = Embedding(input_dim=supplier, output_dim=supplier_emb, name='mf_embedding_item_anbieter',
                                  embeddings_initializer=initializers.random_normal(),
                                  embeddings_regularizer=l2(reg_mf),input_length=1)
    MF_Embedding_Item_mkt = Embedding(input_dim=mkt, output_dim=mkt_emb, name='mf_embedding_item_mkt',
                                  embeddings_initializer=initializers.random_normal(),
                                  embeddings_regularizer=l2(reg_mf), input_length=1)
    
    MF_Embedding_Item_wg = Embedding(input_dim=wgs, output_dim=wgs_emb, name='mf_embedding_item_wg',
                                  embeddings_initializer=initializers.random_normal(),
                                  embeddings_regularizer=l2(reg_mf), input_length=1)

    
    # User
    
    MF_Embedding_User_mkt = Embedding(input_dim=mkt, output_dim=mkt_emb, name='mf_embedding_user_mkt',
                                      embeddings_initializer=initializers.random_normal(),
                                      embeddings_regularizer=l2(reg_mf),
                                      #mask_zero = True,
                                      input_length=1)
    MF_Embedding_User_anbietermkt = Embedding(input_dim=mkt, output_dim=mkt_emb, name='mf_embedding_user_anbietermkt',
                                              embeddings_initializer=initializers.random_normal(),
                                              embeddings_regularizer=l2(reg_mf),
                                              mask_zero = True,
                                              input_length=max_length)
    MF_Embedding_User_wg = Embedding(input_dim=wgs, output_dim=wgs_emb, name='mf_embedding_user_wg',
                                     embeddings_initializer=initializers.random_normal(),
                                     embeddings_regularizer=l2(reg_mf),
                                     mask_zero = True,
                                     input_length=max_length)
    MF_Embedding_User_anbieter = Embedding(input_dim=supplier, output_dim=supplier_emb, name='mf_embedding_user_anbieter',
                                           embeddings_initializer=initializers.random_normal(),
                                           embeddings_regularizer=l2(reg_mf),
                                           mask_zero = True,
                                           input_length=max_length)

    
    # MLP
    #Context
    
    MLP_Embedding_Context_month = Embedding(input_dim=months, output_dim=months_emb, name='mlp_embedding_month',
                                  embeddings_initializer=initializers.random_normal(),
                                  embeddings_regularizer=l2(reg_layers[0]), input_length=1)
        
        
    # Item
    
    MLP_Embedding_Item_anbieter = Embedding(input_dim=supplier, output_dim=supplier_emb, name='mlp_embedding_item_anbieter',
                                  embeddings_initializer=initializers.random_normal(),
                                  embeddings_regularizer=l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item_mkt = Embedding(input_dim=mkt, output_dim=mkt_emb, name='mlp_embedding_item_mkt',
                                  embeddings_initializer=initializers.random_normal(),
                                  embeddings_regularizer=l2(reg_layers[0]),  
                                       input_length=1)
    
    MLP_Embedding_Item_wg = Embedding(input_dim=wgs, output_dim=wgs_emb, name='mlp_embedding_item_wg',
                                  embeddings_initializer=initializers.random_normal(),
                                  embeddings_regularizer=l2(reg_layers[0]), input_length=1)

    
    # User
    
    MLP_Embedding_User_mkt = Embedding(input_dim=mkt, output_dim=mkt_emb, name='mlp_embedding_user_mkt',
                                      embeddings_initializer=initializers.random_normal(),
                                      embeddings_regularizer=l2(reg_layers[0]),
                                      #mask_zero = True,
                                      input_length=1)
    MLP_Embedding_User_anbietermkt = Embedding(input_dim=mkt, output_dim=mkt_emb, name='mlp_embedding_user_anbietermkt',
                                              embeddings_initializer=initializers.random_normal(),
                                              embeddings_regularizer=l2(reg_layers[0]),
                                              mask_zero = True,
                                              input_length=max_length)
    MLP_Embedding_User_wg = Embedding(input_dim=wgs, output_dim=wgs_emb, name='mlp_embedding_user_wg',
                                     embeddings_initializer=initializers.random_normal(),
                                     embeddings_regularizer=l2(reg_layers[0]),
                                     mask_zero = True,
                                     input_length=max_length)
    MLP_Embedding_User_anbieter = Embedding(input_dim=supplier, output_dim=supplier_emb, name='mlp_embedding_user_anbieter',
                                           embeddings_initializer=initializers.random_normal(),
                                           embeddings_regularizer=l2(reg_layers[0]),
                                           #mask_zero = True,
                                           input_length=max_length)

    ## all user ones zb. and then concenate / multiply in MF 
    # MF part
    ## context 
    emb_item_month = Flatten(name = "mf_flat_month")(MF_Embedding_Context_month(month))
    
    ## user
    emb_user_mkt = Flatten(name ="mf_flat_user_mkt")(MF_Embedding_User_mkt(user_mkt))
#     emb_user_anbietermkt = (keras.layers.Lambda(lambda x: mask_aware_mean(x), name = "mf_avg_user_anbietermkt"))(MF_Embedding_User_anbietermkt(user_anbietermkt))
    emb_user_wg = (keras.layers.Lambda(lambda x: mask_aware_mean(x), name = "mf_avg_user_wg"))(MF_Embedding_User_wg(user_wg))
    emb_user_anbieter = (keras.layers.Lambda(lambda x: mask_aware_mean(x), name = "mf_avg_user_anbieter"))(MF_Embedding_User_anbieter(user_anbieter))

    ## item
    emb_item_anbieter = Flatten(name = "mf_flat_item_anbieter")(MF_Embedding_Item_anbieter(item_anbieter))
    emb_item_mkt = Flatten(name = "mf_flat_item_mkt")(MF_Embedding_Item_mkt(item_mkt))
    emb_item_wg = Flatten(name = "mf_flat_item_wg")(MF_Embedding_Item_wg(item_wg))

    # MF connect
    mf_vector = multiply([concatenate([emb_user_mkt, emb_user_wg, emb_user_anbieter, user_preis, user_ve, user_text,
                                       day_online, emb_item_month], name = "mf_user"), 
                          concatenate([emb_item_mkt, emb_item_wg, emb_item_anbieter, item_preis, item_ve, item_text,
                                       day_online, emb_item_month], name = "mf_item")], 
                         name = "mf_multiply")
    
    
    # MLP part
    ## context 
    emb_item_month2 = Flatten(name = "mlp_flat_month")(MLP_Embedding_Context_month(month))
    ## user
    emb_user_mkt2 = Flatten(name ="mlp_flat_user_mkt")(MLP_Embedding_User_mkt(user_mkt))
    emb_user_anbietermkt2 = (keras.layers.Lambda(lambda x: mask_aware_mean(x), name = "mlp_avg_user_anbietermkt"))(MLP_Embedding_User_anbietermkt(user_anbietermkt))
    emb_user_wg2 = (keras.layers.Lambda(lambda x: mask_aware_mean(x), name = "mlp_avg_user_wg"))(MLP_Embedding_User_wg(user_wg))
    emb_user_anbieter2 = (keras.layers.Lambda(lambda x: mask_aware_mean(x), name = "mlp_avg_user_anbieten"))(MLP_Embedding_User_anbieter(user_anbieter))

    ## item
    emb_item_anbieter2 = Flatten(name = "mlp_flat_item_anbieter")(MLP_Embedding_Item_anbieter(item_anbieter))
    emb_item_mkt2 = Flatten(name = "mlp_flat_item_mkt")(MLP_Embedding_Item_mkt(item_mkt))
    emb_item_wg2 = Flatten(name = "mlp_flat_item_wg")(MLP_Embedding_Item_wg(item_wg))

    mlp_vector = concatenate([emb_item_month2, day_online,
                              emb_user_mkt2, emb_user_anbietermkt2, emb_user_wg2, emb_user_anbieter2, user_preis, user_ve, user_text,
                              emb_item_anbieter2, emb_item_mkt2, emb_item_wg2, item_preis, item_ve, item_text], name = "mlp_conc")

    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers[idx]), activation=lrelu, name="layer%d" % idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP parts
    predict_vector = concatenate([mf_vector, mlp_vector])

    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer=initializers.lecun_normal(),
                       name="prediction")(predict_vector)

    model_ = Model(inputs=[month, day_online,
                           item_anbieter, item_mkt, item_wg, item_preis, item_ve, user_text,
                           user_mkt, user_anbietermkt, user_wg, user_anbieter, user_preis, user_ve, item_text],
                   outputs=prediction)

    return model_