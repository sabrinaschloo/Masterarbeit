#### get all clicks and use to add new items and update user history ####
# only add items that have been cicked - and everyday add all items that api was requested to evaluate but didnt find in db ! # 
## packages
import pandas as pd
import numpy as np 
import mysql.connector as mariadb
import functools 
import pickle
from bs4 import BeautifulSoup
from datetime import datetime
import psycopg2
from sqlalchemy import create_engine, MetaData, Table
import sqlalchemy as db


###### Db Connection ###### 
# production 
pre = "/update/" 
import user as user
import item as item
person_recommender_db = ###
mariadb_str = ###
# maria_db
rec_db = ###


#### Add new Users  ####
## get users from Userstammdaten with reg > yesterday
# load last erstRegDatum
with open (pre + 'preprocessing/max_regdate.pkl', 'rb') as f: ## needs change
    max_reg = pickle.load(f)
#max_reg = "2020-01-21 23:59:41"

# connect to stastic db
mariadb_connection = mariadb.connect('###') 
cursor_maria = mariadb_connection.cursor(buffered=True)

# select users
new_users = user.get_new_users(cursor_maria, max_reg)

print('%s new users found' % (len(new_users),))

# Encode usermkt 
new_users.loc[:, 'usermkt_enc'] = user.encode_mkt(new_users['usermkt'])

# add 0 in all fields that we have
new_user_done = user.fill_empty(new_users)

# find max date to save later
new_max_reg = max(new_users.regdate) #'2019-12-16 03:41:18'

# drop fields i don't need
new_user_done.drop(['regdate', 'usermkt'], axis = 1, inplace = True)

# prepare
new_user_save = new_user_done[['anbieterid_enc_user', 'datum_click', 'clicked_before',
    'anbietermarktplatz_enc_user', 'warengruppe_enc_user', 'text_vec_user',
    'preis_std_user', 'minve_std_user', 'preis_log_std_user',
    'minve_log_std_user', 'usermkt_enc', 'userid']]
new_users_save_list = new_user_save.to_records(index = False, column_dtypes = {'anbieterid_enc_user' : 'object', 'datum_click' : 'datetime64[s]', 'clicked_before' : 'object',
    'anbietermarktplatz_enc_user' : 'object', 'warengruppe_enc_user' : 'object', 'text_vec_user' : 'object',
    'preis_std_user' : 'float', 'minve_std_user' : 'float', 'preis_log_std_user' : 'float',
    'minve_log_std_user' : 'float', 'usermkt_enc': 'int', 'userid' : 'object'}).tolist()

# connect to postgres
conn_rec = psycopg2.connect(person_recommender_db)

# save
user.insert_user_initial_task(conn_rec, new_users_save_list) # getestet, funktioniert !!!

#new_user_done.to_sql('user_enc', engine, index = None, if_exists = 'append')

with open (pre + 'preprocessing/max_regdate.pkl', 'wb') as f:
    pickle.dump(new_max_reg, f, protocol = 2)

print('%s new users added to db' % (len(new_user_done),))


#### Get Data zu Update based on Clicks ####

conn_rec = psycopg2.connect(person_recommender_db)
mariadb_connection = mariadb.connect('###')
cursor_maria = mariadb_connection.cursor(buffered=True)

last_click = pd.read_sql("select MAX(datum_last_click) from item_features_raw", conn_rec) # fehlt noch das genaue herausholen des Datums !! 
last_click = last_click['max'][0]
#last_click = '2020-01-18 23:59:57'

clicks = item.get_new_clicks(cursor_maria, last_click)

# drop duplicates to get data that needs update
update_items = clicks.drop_duplicates('anbieter_artikelnummer', keep = "last")
update_users = clicks.drop_duplicates('userid', keep = "last")

print('%s items to update' % (len(update_items), ))
print('%s user to update' % (len(update_users), ))


########## Update items in db ##########
### add/update items that are in click stream 
# save all clicked items new / update to always have current data !! 
### create raw item features
update_items.reset_index(inplace= True, drop = True)

# Fill NaN warengruppe and calculate preis_euro
update_items.loc[:, 'warengruppe'] = update_items.warengruppe.fillna("NN")
update_items.loc[:,'preis_euro'] = item.price_to_euro(update_items)

# extract text 
update_items.loc[:,'text_all'] = update_items.bezeichnung + ' ' + update_items.beschreibung

beschreibung = []
for h in update_items['text_all']:
    try:
        b = BeautifulSoup(h, features = "html.parser")
        beschreibung.append(b.get_text())
    except:
        beschreibung.append(None)

update_items.loc[:,'text_clean'] = beschreibung
print("raw features generate")

# select only needed columns
update_items = update_items[['datum_last_click', 'anbieterid','anbietermarktplatz', 'erstanlagedatum', 'stueck_pro_ve', 'warengruppe', 'preis_euro', 'text_clean', 'anbieter_artikelnummer']]

### encode item features and save to item_enc
## encode
update_items_trans = item.transform_data (update_items)

## Save encoded data to item_enc
# prepare data for saving
update_items_trans = update_items_trans[['anbieterid_enc', 'anbietermarktplatz_enc','warengruppe_enc', 'text_vec', 'preis_std', 'minve_std',  'minve_log_std', 'preis_log_std', 'erstanlagedatum', 'anbieter_artikelnummer']]
update_items_trans_final = update_items_trans.merge(update_items_trans, how = "left", on = 'anbieter_artikelnummer', suffixes = ('', '2'))
items_enc_save_list = update_items_trans_final.to_records(index = False, column_dtypes = {'anbieterid_enc' : 'int', 'anbietermarktplatz_enc' : 'int','warengruppe_enc' : 'int', 'text_vec' : 'object', 'preis_std': 'float', 
        'minve_std': 'float', 'minve_log_std': 'float', 'preis_log_std': 'float', 'erstanlagedatum': 'datetime64[s]', 'anbieter_artikelnummer': 'object', 'anbieterid_enc2' : 'int', 'anbietermarktplatz_enc2' : 'int',
        'warengruppe_enc2' : 'int', 'text_vec2' : 'object', 'preis_std2': 'float', 'minve_std2': 'float', 'minve_log_std2': 'float', 'preis_log_std2': 'float', 'erstanlagedatum2': 'datetime64[s]'}).tolist()
# connect to db
conn_rec = psycopg2.connect(person_recommender_db)
# save
item.update_item_enc_task(conn_rec, items_enc_save_list) # getestet, funktioniert !!! 
print("all items updated")




######### Update users in db ###########
engine_rec_db = create_engine(rec_db)
mariadb_connection = mariadb.connect('###')
cursor_maria = mariadb_connection.cursor(buffered=True)

update_users.reset_index (drop = True, inplace = True)
# fill missing usermarketplaces with EU
update_users.loc[:, 'usermkt'] = update_users.usermkt.fillna("EU")

#### get last 200 clicks for every user, get encoded item info, create user profile and save to db 
# do in batches so data is available online earlier
batch_size = 500
for batch in (range(0, len(update_users), batch_size)):
    # select batch
    user_batch = update_users[batch:batch + batch_size]
    user_batch.reset_index (inplace = True, drop = True)
    # get profile for each user
    user_trans_list = []
    for n in range(len(user_batch)):
        user_transformed = user.get_clicks_aggregate(cursor_maria, engine_rec_db, user_batch.userid[n], user_batch.datum_last_click[n], user_batch.usermkt[n])
        user_trans_list.append(user_transformed)
    # combine all user profiles
    user_trans_df = pd.concat(user_trans_list)
    # prepare for saving
    user_trans_df = user_trans_df[['anbieterid_enc_user', 'datum_click', 'clicked_before',
        'anbietermarktplatz_enc_user', 'warengruppe_enc_user', 'text_vec_user',
        'preis_std_user', 'minve_std_user', 'preis_log_std_user',
        'minve_log_std_user', 'usermkt_enc', 'userid']]
    user_trans_df_final = user_trans_df.merge(user_trans_df, how = 'left', on = 'userid', suffixes = ("", "2"))
    users_enc_save_list = user_trans_df_final.to_records(index = False, column_dtypes = {'anbieterid_enc_user' : 'object', 'datum_click' : 'datetime64[s]', 'clicked_before' : 'object',
        'anbietermarktplatz_enc_user' : 'object', 'warengruppe_enc_user' : 'object', 'text_vec_user' : 'object',
        'preis_std_user' : 'float', 'minve_std_user' : 'float', 'preis_log_std_user' : 'float',
        'minve_log_std_user' : 'float', 'usermkt_enc': 'int', 'userid' : 'object', 
        'anbieterid_enc_user2' : 'object', 'datum_click2' : 'datetime64[s]', 'clicked_before2' : 'object',
        'anbietermarktplatz_enc_user2' : 'object', 'warengruppe_enc_user2' : 'object', 'text_vec_user2' : 'object',
        'preis_std_user2' : 'float', 'minve_std_user2' : 'float', 'preis_log_std_user2' : 'float',
        'minve_log_std_user2' : 'float', 'usermkt_enc2': 'int'}).tolist()
    # connect to db       
    conn_rec = psycopg2.connect(person_recommender_db)
    # save
    user.update_user_enc_task(conn_rec, users_enc_save_list) # getestet, funktioniert !!!
    print("User batch %s transformed and saved" % (batch, ))
print("all users updated")



### Save data to item_features_raw - do last so it does not stop user profiles from being updated
# prepare data for saving
update_items_final = update_items.merge(update_items, how = "left", on = 'anbieter_artikelnummer', suffixes = ('', '2'))
items_save_list = update_items_final.to_records(index = False, column_dtypes = {'datum_last_click' : 'datetime64[s]', 'anbieterid' : 'int', 'anbietermarktplatz' : 'object', 'erstanlagedatum' : 'datetime64[s]', 
        'stueck_pro_ve': 'int', 'warengruppe': 'object', 'preis_euro' : 'float', 'text_clean' : 'object', 'anbieter_artikelnummer' : 'object', 'datum_last_click2' : 'datetime64[s]', 'anbieterid2' : 'int', 
        'anbietermarktplatz2' : 'object', 'erstanlagedatum2' : 'datetime64[s]', 'stueck_pro_ve2': 'int', 'warengruppe2': 'object', 'preis_euro2' : 'float', 'text_clean2' : 'object'} ).tolist()
# connect to db
conn_rec = psycopg2.connect(person_recommender_db)
# save
item.update_item_raw_task(conn_rec, items_save_list) # getestet, funktioniert !!!

print("all items raw saved")