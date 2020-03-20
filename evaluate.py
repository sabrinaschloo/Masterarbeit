import numpy as np
import pandas as pd
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import pickle

def unpickle_data(data):
    df = data.copy()
    columns = ['clicked_before', 'text_vec', 'anbieterid_enc_user', 'anbietermarktplatz_enc_user', 'warengruppe_enc_user', 'text_vec_user']
    for column in columns:
        try:
            df[column] = [pickle.loads(df.loc[i,column]) for i in range(len(df))]
        except KeyError:
            pass
    return (df)



# transform data 
def transform(data, max_length):
    #data_bundle = data_bundle.dropna().reset_index()
    data_bundle = unpickle_data(data)
    # context
    month = np.nan_to_num(data_bundle.month_enc.values.reshape(-1,1))
    days_online = np.nan_to_num(data_bundle.days_online_std.values.reshape(-1,1))#, dtype = "float32")
    #days_online = np.nan_to_num(data_bundle.days_online_log_std.values.reshape(-1,1))#, dtype = "float32")

    # item
    item_anbieter = np.nan_to_num(data_bundle.anbieterid_enc.values.reshape(-1,1)) # nur weil es noch Nan gab - fix ich noch
    item_mkt= np.nan_to_num(data_bundle.anbietermarktplatz_enc.values.reshape(-1,1))
    item_wg = np.nan_to_num(data_bundle.warengruppe_enc.values.reshape(-1,1))
    item_preis = np.nan_to_num(data_bundle.preis_std.values.reshape(-1,1))
    item_ve = np.nan_to_num(data_bundle.minve_std.values.reshape(-1,1))
    #item_preis = np.nan_to_num(data_bundle.preis_log_std.values.reshape(-1,1))
    #item_ve = np.nan_to_num(data_bundle.minve_log_std.values.reshape(-1,1))
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
    user_preis = np.nan_to_num(data_bundle.preis_std_user.values.reshape(-1,1))
    user_ve = np.nan_to_num(data_bundle.minve_std_user.values.reshape(-1,1))
    #user_preis = np.nan_to_num(data_bundle.preis_log_std_user.values.reshape(-1,1))
    #user_ve = np.nan_to_num(data_bundle.minve_log_std_user.values.reshape(-1,1))

    # features 
    x_values = [month, days_online, item_anbieter, item_mkt, item_wg, item_preis, item_ve, item_text, user_mkt, user_anbietermkt, user_wg, user_anbieter, user_preis, user_ve, user_text]
    return(x_values)


def transform_log(data, max_length):
    data_bundle = unpickle_data(data)
    #data_bundle = data_bundle.dropna().reset_index()
    # context
    month = np.nan_to_num(data_bundle.month_enc.values.reshape(-1,1))
    days_online = np.nan_to_num(data_bundle.days_online_log_std.values.reshape(-1,1))#, dtype = "float32")

    # item
    item_anbieter = np.nan_to_num(data_bundle.anbieterid_enc.values.reshape(-1,1)) # nur weil es noch Nan gab - fix ich noch
    item_mkt= np.nan_to_num(data_bundle.anbietermarktplatz_enc.values.reshape(-1,1))
    item_wg = np.nan_to_num(data_bundle.warengruppe_enc.values.reshape(-1,1))
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
    user_preis = np.nan_to_num(data_bundle.preis_log_std_user.values.reshape(-1,1))
    user_ve = np.nan_to_num(data_bundle.minve_log_std_user.values.reshape(-1,1))

    # features 
    x_values = [month, days_online, item_anbieter, item_mkt, item_wg, item_preis, item_ve, item_text, user_mkt, user_anbietermkt, user_wg, user_anbieter, user_preis, user_ve, user_text]
    return(x_values)

## calc AUROC per user 
# do outside: 
#users = pd.read_sql("SELECT DISTINCT userID from target_testing_enc", engine).values.tolist()
#for u in users:
#   data_u = pd.read_sql("SELECT * from target_testing_enc where userID = %s", engine, params = (u,))

def ROC_per_User (model, data_u):
    if len(data_u.pick.unique()) > 1 :
        # transform data
        data_u_p = transform(data_u, 200)
        # select truth and make prediction
        y_true = data_u.pick.values.reshape(-1,1)
        y_score = model.predict(x = data_u_p)
        # calc AUROC
        roc = roc_auc_score(y_true, y_score)
    else: 
        roc = None
    return (roc)

# Calc top k precision per User
def Prec_at_k (model, data_u, k):    
    # transform for model
    data_u_p = transform(data_u, 200)
    # select truth and make prediction
    data_u['y_score'] = model.predict(x = data_u_p)
    data_u_top = data_u.sort_values('y_score', ascending = False)[:k]
    y_true = data_u_top.pick.values.reshape(-1,1)
    y_score = data_u_top.y_score.values.reshape(-1,1)
    # calc AUROC
    prec = precision_score(y_true, y_score.round())
    return (prec)


# Calc top k recall per User
def Recall_at_k (model, data_u, k):    
    # transform for model
    data_u_p = transform(data_u, 200)
    # select truth and make prediction
    data_u['y_score'] = model.predict(x = data_u_p)
    data_u_top = data_u.sort_values('y_score', ascending = False)[:k]
    y_true = data_u_top.pick.values.reshape(-1,1)
    y_score = data_u_top.y_score.values.reshape(-1,1)
    # calc AUROC
    recall = recall_score(y_true, y_score.round())
    return (recall)