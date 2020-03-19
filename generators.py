import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import numpy as np

max_length = 200

# needed for sqlite - need to unpickle all arrays in db
def unpickle_data(df):
    df = data.copy()
    columns = ['clicked_before', 'text_vec', 'anbieterid_enc_user', 'anbietermarktplatz_enc_user', 'warengruppe_enc_user', 'text_vec_user']
    for column in columns:
        try:
            df[column] = [pickle.loads(df.loc[i,column]) for i in range(len(df))]
        except KeyError:
            pass
    return (df)


# Functions for generator of batch and validation data when want to shuffle input data, use fixed continuous data
def generate_batches_shuffle_new(engine, batch_size, train_list):
    while True:
           for batch in range(0,len(train_list), batch_size):
                batch_items = tuple(train_list[int(batch): int(batch+batch_size)])
                #start = int(batch)
                #end = int(batch + batch_size)
                # get batch from db
                query = f"""
                    SELECT * FROM target_training_enc where index IN {batch_items}
                """
                data_bundle_db = pd.read_sql_query(query, engine)
                # needed for sqlite
                data_bundle = unpickle_data(data_bundle_db)
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

                # target
                y_train = data_bundle.pick.values.reshape(-1,1)

                yield ([month, days_online,
                       item_anbieter, item_mkt, item_wg, item_preis, item_ve, item_text,
                       user_mkt, user_anbietermkt, user_wg, user_anbieter, user_preis, user_ve, user_text], y_train)




