import numpy as np
import tensorflow as tf
from numpy.random import seed
from tensorflow import set_random_seed
seed(1)
set_random_seed(1)

from keras import initializers
from keras.regularizers import l2
from keras.models import Model, model_from_json
from keras.layers import Embedding, Input, Dense, merge, Flatten, concatenate, multiply, dot, Reshape, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing.sequence import pad_sequences
import keras.callbacks
from keras import backend as K
import keras
from datetime import datetime
import pickle
import pandas as pd

std_days_online = pickle.load (open( "/api/model/preprocessing/new_scaler_days_online_log.pkl", "rb" ) )
#std_days_online = pickle.load (open( "model/preprocessing/scaler_days_online.pkl", "rb" ) )


def rank_list(model, engine, userid, df_in):
    data_u = pd.read_sql("SELECT * FROM user_enc where userid = %(user)s ", engine, params = {'user' : str(userid)})
    # send error message if user not in db yet
    if len(data_u) == 0:
        df =  df_in.copy()
        df.loc[:,'score'] = 0
        j = df.groupby(["position"]).apply(lambda x: x[["anbieter_artikelnummer", "score"]].to_dict(orient='records')).to_dict()
        #j = ({ "message" : "User not available"}, 500)
    else:
        ### try this new pd. function
        #query = "select * from TABLENAME"
        #df = pd.read_sql_query(query, sql_engine)
        data_i = pd.read_sql("SELECT * FROM item_enc where anbieter_artikelnummer IN %(items)s ", engine, params = {'items' : tuple(df_in.anbieter_artikelnummer.tolist())})
        ### join data
        # add userID for join and join
        data_i.loc[:,'userid'] = str(userid)
        df = data_i.merge(data_u, on = "userid")
        # transform data and make prediction
        df_trans = transform(df, 200)
        df.loc[:,'score'] = model.predict(x = df_trans)
        df_score = df[['anbieter_artikelnummer', 'score']]
        # merge results
        df_score = pd.merge(df_in, df_score, how = 'left', on = 'anbieter_artikelnummer').fillna(0)
        # create return json
        j = df_score.groupby(["position"]).apply(lambda x: x[["anbieter_artikelnummer", "score"]].to_dict(orient='records')).to_dict()
    return (j)



#### function to get data in right format for chosen Model 
def transform(data_bundle_in, max_length):
    data_bundle = unpickle_data(data_bundle_in)
    # context
    month = np.repeat(datetime.now().month, len(data_bundle)).reshape(-1,1)
    online = (datetime.now() - data_bundle.erstanlagedatum)
    days_online_unsc = [online[i].days for i in range(len(online))]
    days_online_log = np.array([np.log(days_online_unsc[i] +1e-2 ) for i in range(len(online))]).reshape(-1,1)
    days_online = std_days_online.transform(days_online_log)
    # item
    item_anbieter = np.nan_to_num(data_bundle.anbieterid_enc.values.reshape(-1,1)) # nur weil es noch Nan gab - fix ich noch
    item_mkt= np.nan_to_num(data_bundle.anbietermarktplatz_enc.values.reshape(-1,1))
    item_wg = np.nan_to_num(data_bundle.warengruppe_enc.values.reshape(-1,1))
    #item_preis = np.nan_to_num(data_bundle.preis_std.values.reshape(-1,1))
    #item_ve = np.nan_to_num(data_bundle.minve_std.values.reshape(-1,1))
    item_preis = np.nan_to_num(data_bundle.preis_log_std.values.reshape(-1,1))
    item_ve = np.nan_to_num(data_bundle.minve_log_std.values.reshape(-1,1))
    list_text = []
    list_text_user = []
    for i in range(len(data_bundle)):
        list_text.append(data_bundle.text_vec[i])
        list_text_user.append(data_bundle.text_vec_user[i])
    item_text = np.array(list_text, ndmin = 2)
    # user
    user_text = np.array(list_text_user, ndmin = 2)
    user_mkt = np.nan_to_num(data_bundle.usermkt_enc.values.reshape(-1,1))
    user_anbieter = pad_sequences(data_bundle.anbieterid_enc_user, maxlen = max_length, padding = "pre")
    user_anbietermkt = pad_sequences(data_bundle.anbietermarktplatz_enc_user, maxlen = max_length, padding = "pre")
    user_wg = pad_sequences(data_bundle.warengruppe_enc_user, maxlen = max_length, padding = "pre")
    #user_preis = np.nan_to_num(data_bundle.preis_std_user.values.reshape(-1,1))
    #user_ve = np.nan_to_num(data_bundle.minve_std_user.values.reshape(-1,1))
    user_preis = np.nan_to_num(data_bundle.preis_log_std_user.values.reshape(-1,1))
    user_ve = np.nan_to_num(data_bundle.minve_log_std_user.values.reshape(-1,1))
    # features 
    x_values = [month, days_online, item_anbieter, item_mkt, item_wg, item_preis, item_ve, item_text, user_mkt, user_anbietermkt, user_wg, user_anbieter, user_preis, user_ve, user_text]
    return(x_values)


def unpickle_data(df):
    df = data.copy()
    columns = ['clicked_before', 'text_vec', 'anbieterid_enc_user', 'anbietermarktplatz_enc_user', 'warengruppe_enc_user', 'text_vec_user']
    for column in columns:
        try:
            df[column] = [pickle.loads(df.loc[i,column]) for i in range(len(df))]
        except KeyError:
            pass
    return (df)