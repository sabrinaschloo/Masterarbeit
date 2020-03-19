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


###### Db Connection ###### 
# test
import app.update.functions as functions
person_recommender_db = "host='host.docker.internal' port='5431' dbname='person_recommender' user='postgres' password='power2020'"
rec_db = 'postgres+psycopg2://postgres:power2020@host.docker.internal:5431/person_recommender'
mariadb_connection = mariadb.connect(host='host.docker.internal',port=3306,user='bigdata',password='Byckd4t4',database='bigdata')
pre = "app/update/" # auch in funktionen ändern ! 
#pre = ""
# get clicks later than last update
last_click = '2019-09-16 04:00:00' # ende der Trainingsdaten ! 


# production 
#import functions as functions
#person_recommender_db = "host='co1.db.schimmel' port='5432' dbname='person_recommender' user='postgres' password='power2020'"
#rec_db = 'postgres+psycopg2://postgres:power2020@co1.db.schimmel:5432/person_recommender'
#mariadb_connection = mariadb.connect(host='192.168.3.114',port=3306,user='bigdata',password='Byckd4t4',database='bigdata') # getestet - funktioniert
#pre = "/update/" # auch in funktionen ändern ! oder ""
# get clicks later than last update
#last_click = pd.read_sql("select MAX(datum_last_click) from item_features_raw", conn_rec) # fehlt noch das genaue herausholen des Datums !! 
#last_click = last_click['max'][0]

cursor_maria = mariadb_connection.cursor(buffered=True)
conn_rec = psycopg2.connect(person_recommender_db)


## vllt auch einfach 
items_unique = pd.read_sql ('SELECT anbieter_artikelnummer, max(datum) from artikelklicks_current where datum > "2019-09-16 04:00:00" group by anbieter_artikelnummer', mariadb_connection)

def update_item_enc_task(conn, task):
    """
    update all parameters in item_enc
    :param conn:
    :param task:
    """
    sql = ''' INSERT INTO item_enc (anbieterid_enc, anbietermarktplatz_enc, warengruppe_enc, text_vec, preis_std, minve_std,  minve_log_std, preis_log_std, erstanlagedatum,anbieter_artikelnummer) 
                VAlUES (%s, %s, %s, %s, %s, %s, %s, %s,%s, %s)
                ON CONFLICT (anbieter_artikelnummer)
                DO 
                    UPDATE
                    SET anbieterid_enc = %s, anbietermarktplatz_enc = %s,warengruppe_enc = %s, text_vec = %s, preis_std = %s, minve_std = %s,  minve_log_std = %s, preis_log_std = %s, erstanlagedatum = %s'''
    cur = conn.cursor()
    cur.executemany(sql, task) # task is list of mulitple updates
    conn.commit()

def update_item_raw_task(conn, task):
    """
    update all features in item_features_raw
    :param conn:
    :param task:
    """
    sql = ''' INSERT INTO item_features_raw (datum_last_click, anbieterid, anbietermarktplatz, erstanlagedatum, stueck_pro_ve, warengruppe, preis_euro, text_clean, anbieter_artikelnummer)
                VALUES (%s, %s, %s, %s, %s, %s,%s, %s, %s)
                ON CONFLICT (anbieter_artikelnummer)
                DO 
                UPDATE
                SET datum_last_click = %s, anbieterid = %s, anbietermarktplatz = %s, erstanlagedatum = %s, stueck_pro_ve = %s, warengruppe = %s, preis_euro = %s, text_clean = %s '''
    
    cur = conn.cursor()
    cur.executemany(sql, task) # task is list of mulitple updates
    conn.commit()

def update_user_enc_task(conn, task):
    """
    update all parameters in item_enc
    :param conn:
    :param task:
    """
    sql = ''' INSERT INTO user_enc (anbieterid_enc_user, datum_click, clicked_before, anbietermarktplatz_enc_user, warengruppe_enc_user, text_vec_user,  preis_std_user, minve_std_user, preis_log_std_user, minve_log_std_user, usermkt_enc, userid) 
                VALUES (%s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s) 
                ON CONFLICT (userid)
                DO 
                UPDATE
                SET anbieterid_enc_user = %s, datum_click = %s, clicked_before = %s, anbietermarktplatz_enc_user = %s, warengruppe_enc_user = %s, text_vec_user = %s,  preis_std_user = %s, minve_std_user = %s, preis_log_std_user = %s, minve_log_std_user = %s, usermkt_enc = %s'''
   
    cur = conn.cursor()
    cur.executemany(sql, task) # task is list of mulitple updates
    conn.commit()



print('%s items to update' % (len(items_unique), ))
## first test the extraction from db !!!
batch_size = 5000
for batch in range(0, len(items_unique), batch_size):
    artikelnummer = tuple(items_unique[batch:batch+batch_size].anbieter_artikelnummer.values.tolist())
    #print(artikelnummer)
    query = f"""SELECT i.anbieter_artikelnummer, i.artikelID, i.erstanlageDatum, i.waehrung, i.stueckpreis, i.stueck_pro_ve, i.anbieterID,
        a.anbietermarktplatz,
        w.warengruppe,
        t.bezeichnung, t.beschreibung
        FROM artikelstammdaten as i
        LEFT JOIN anbieterstammdaten as a
        ON i.anbieterID = a.anbieterID
        LEFT JOIN artikelwarengruppe as w
        ON i.artikelID = w.artikelID
        LEFT JOIN artikeltexte as t
        ON i.artikelID = t.artikelID 
        WHERE i.marktplatz = 'EU' and anbieter_artikelnummer IN {artikelnummer} """
    data = pd.read_sql_query(query, mariadb_connection)
    #print(data)
    data = data.sort_values(by = "artikelID", ascending = True)
    update_items = data.drop_duplicates('anbieter_artikelnummer', keep = "last") 
    ### either continue or append to list
    update_items.reset_index(inplace= True, drop = True)
    update_items.loc[:,'stueckpreis'] = update_items['stueckpreis'].astype(float)
    update_items['exchangeRate'] = np.where(update_items['waehrung'] == 'EUR', float(1) , 0)
    update_items['exchangeRate'] = np.where(update_items['waehrung'] == 'PLN', float(0.228786) , update_items['exchangeRate'] )
    update_items['exchangeRate'] = np.where(update_items['waehrung'] == 'HUF', float(0.00299247), update_items['exchangeRate'] )
    update_items['preis_euro'] = update_items['stueckpreis'] * update_items['exchangeRate']
    # extract text 
    update_items['text_all'] = update_items.bezeichnung + ' ' + update_items.beschreibung
    beschreibung = []
    for h in update_items['text_all']:
        try:
            b = BeautifulSoup(h, features = "html")
            beschreibung.append(b.get_text())
        except:
            beschreibung.append(None)
    update_items['text_clean'] = beschreibung
    update_items['datum_last_click'] = datetime.now()
    update_items.columns = ['anbieter_artikelnummer', 'artikelid', 'erstanlagedatum', 'waehrung',
       'stueckpreis', 'stueck_pro_ve', 'anbieterid', 'anbietermarktplatz',
       'warengruppe', 'bezeichnung', 'beschreibung',
       'exchangeRate', 'preis_euro', 'text_all', 'text_clean', 'datum_last_click']
    # create data to save
    update_items = update_items[['datum_last_click', 'anbieterid','anbietermarktplatz', 'erstanlagedatum', 'stueck_pro_ve', 'warengruppe', 'preis_euro', 'text_clean', 'anbieter_artikelnummer']]
    update_items_final = update_items.merge(update_items, how = "left", on = 'anbieter_artikelnummer', suffixes = ('', '2'))
    items_save_list = update_items_final.to_records(index = False, column_dtypes = {'datum_last_click' : 'datetime64[s]', 'anbieterid' : 'int', 'anbietermarktplatz' : 'object', 'erstanlagedatum' : 'datetime64[s]', 'stueck_pro_ve': 'int', 'warengruppe': 'object', 'preis_euro' : 'float', 'text_clean' : 'object', 'anbieter_artikelnummer' : 'object', 'datum_last_click2' : 'datetime64[s]', 'anbieterid2' : 'int', 'anbietermarktplatz2' : 'object', 'erstanlagedatum2' : 'datetime64[s]', 'stueck_pro_ve2': 'int', 'warengruppe2': 'object', 'preis_euro2' : 'float', 'text_clean2' : 'object'} ).tolist()
    # connect to db
    conn_rec = psycopg2.connect(person_recommender_db)
    # save
    update_item_raw_task(conn_rec, items_save_list) # getestet, funktioniert !!!
    ### Transform data 
    update_items_trans = functions.transform_data (update_items)
    ## Save encoded data to item_enc
    update_items_trans = update_items_trans[['anbieterid_enc', 'anbietermarktplatz_enc','warengruppe_enc', 'text_vec', 'preis_std', 'minve_std',  'minve_log_std', 'preis_log_std', 'erstanlagedatum', 'anbieter_artikelnummer']]
    update_items_trans_final = update_items_trans.merge(update_items_trans, how = "left", on = 'anbieter_artikelnummer', suffixes = ('', '2'))
    items_enc_save_list = update_items_trans_final.to_records(index = False, column_dtypes = {'anbieterid_enc' : 'int', 'anbietermarktplatz_enc' : 'int','warengruppe_enc' : 'int', 'text_vec' : 'object', 'preis_std': 'float', 'minve_std': 'float', 'minve_log_std': 'float', 'preis_log_std': 'float', 'erstanlagedatum': 'datetime64[s]', 'anbieter_artikelnummer': 'object', 'anbieterid_enc2' : 'int', 'anbietermarktplatz_enc2' : 'int','warengruppe_enc2' : 'int', 'text_vec2' : 'object', 'preis_std2': 'float', 'minve_std2': 'float', 'minve_log_std2': 'float', 'preis_log_std2': 'float', 'erstanlagedatum2': 'datetime64[s]'}).tolist()
    # connect to db
    conn_rec = psycopg2.connect(person_recommender_db)
    # save
    update_item_enc_task(conn_rec, items_enc_save_list) # getestet, funktioniert !!! 
    print("Item batch %s transformed and saved" % (batch, ))



### Test ###
# ist in items but not in db... 
## did not add items that where not previously in db.. 
# query_t = f"SELECT * from item_features_raw where anbieter_artikelnummer in {artikelnummer}"
# already_done = pd.read_sql_query(query_t, conn_rec)
# len(already_done)
# len(artikelnummer)
# #update_item_enc_task(conn_rec, items_enc_save_list)
# update_items_trans_test = update_items_trans.merge(update_items_trans, how = "left", on = 'anbieter_artikelnummer', suffixes = ('', '2'))

# items_enc_save_list_test = update_items_trans_test.to_records(index = False, column_dtypes = {'anbieterid_enc' : 'int', 'anbietermarktplatz_enc' : 'int','warengruppe_enc' : 'int', 'text_vec' : 'object', 'preis_std': 'float', 'minve_std': 'float', 'minve_log_std': 'float', 'preis_log_std': 'float', 'erstanlagedatum': 'datetime64[s]', 'anbieter_artikelnummer': 'object', 'anbieterid_enc2' : 'int', 'anbietermarktplatz_enc2' : 'int','warengruppe_enc2' : 'int', 'text_vec2' : 'object', 'preis_std2': 'float', 'minve_std2': 'float', 'minve_log_std2': 'float', 'preis_log_std2': 'float', 'erstanlagedatum2': 'datetime64[s]'}).tolist()

# update_item_enc_task(conn_rec, items_enc_save_list_test)
#pd.read_sql("SELECT * from item_enc where anbieter_artikelnummer = '00690052K1230Z'", conn_rec)

#pd.read_sql("SELECT * from artikelstammdaten where anbieter_artikelnummer = '00690052K1230Z' ", mariadb_connection)

# # raw 

# update_items_final = update_items.merge(update_items, how = "left", on = 'anbieter_artikelnummer', suffixes = ('', '2'))
# items_save_list = update_items_final.to_records(index = False, column_dtypes = {'datum_last_click' : 'datetime64[s]', 'anbieterid' : 'int', 'anbietermarktplatz' : 'object', 'erstanlagedatum' : 'datetime64[s]', 'stueck_pro_ve': 'int', 'warengruppe': 'object', 'preis_euro' : 'float', 'text_clean' : 'object', 'anbieter_artikelnummer' : 'object', 'datum_last_click2' : 'datetime64[s]', 'anbieterid2' : 'int', 'anbietermarktplatz2' : 'object', 'erstanlagedatum2' : 'datetime64[s]', 'stueck_pro_ve2': 'int', 'warengruppe2': 'object', 'preis_euro2' : 'float', 'text_clean2' : 'object'} ).tolist()
# update_item_raw_task(conn_rec, items_save_list)






######### Update users in db ###########
update_users = pd.read_sql ('SELECT cl.userID, max(cl.datum), u.erstRegMarktplatz from artikelklicks_current as cl left join userstammdaten as u on cl.userID = u.userID where datum > "2019-09-16 04:00:00" group by userID', mariadb_connection)


update_users.columns = ['userid', 'datum_last_click', 'usermkt']
print('%s user to update' % (len(update_users), ))
update_users = update_users[update_users.userid > 0]
update_users.loc[:, 'usermkt'] = update_users.usermkt.fillna("EU")


engine_rec_db = create_engine(rec_db)
cursor_maria = mariadb_connection.cursor(buffered=True)
update_users.reset_index (drop = True, inplace = True)
# get last 200 clicks for every user, get encoded item info and create user profile

#user_batch = update_users[:5]

## muss funktion noch laden !!! 
batch_size = 1000
for batch in (range(1000, len(update_users), batch_size)):
    user_batch = update_users[batch:batch + batch_size].copy()
    user_batch.reset_index (inplace = True, drop = True)
    user_trans_list = []
    for n in range(len(user_batch)):
        user_transformed = get_clicks_aggregate(cursor_maria, engine_rec_db, user_batch.userid[n], user_batch.datum_last_click[n], user_batch.usermkt[n])
        user_trans_list.append(user_transformed)
    # combine all user profiles
    user_trans_df = pd.concat(user_trans_list)
    # Save data to user_enc
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
    update_user_enc_task(conn_rec, users_enc_save_list) #getestet, funktioniert !!!
    print("User batch %s transformed and saved" % (batch, ))
print("all users updated")



### Nochmal machen für alle User , die preis_std_user = 0 und preis_std_log_user != 0 - sind 60.000 - get out usermkt_enc und userID !! 

update_users = pd.read_sql ('SELECT userid, datum_click, usermkt_enc from user_enc where preis_std_user = 0 and preis_log_std_user > 0', conn_rec)


update_users.columns = ['userid', 'datum_last_click', 'usermkt']
print('%s user to update' % (len(update_users), ))
#update_users.loc[:, 'usermkt'] = update_users.usermkt.fillna("EU")


engine_rec_db = create_engine(rec_db)
cursor_maria = mariadb_connection.cursor(buffered=True)
update_users.reset_index (drop = True, inplace = True)
# get last 200 clicks for every user, get encoded item info and create user profile

#user_batch = update_users[:5]
# Kleine Änderung hier - da usuermkt schon encoded ist 

def get_clicks_aggregate (cursor_maria, engine_rec_db, userid, date, usermkt): 
    # transform usermkt ##############
    mkt_user = usermkt
    # get last 200 clicks 
    q_update_user = """SELECT DISTINCT anbieter_artikelnummer from artikelklicks_current where userID = %s ORDER BY datum DESC LIMIT 200"""
    items_clicked = get_user_clicks(cursor_maria, q_update_user, userid)
    items_clicked_last200 = items_clicked.values.flatten().tolist()
    if len(items_clicked_last200) > 1:
    # try to get encoded item data
        try: 
            query = f"""SELECT * from item_enc WHERE anbieter_artikelnummer IN {tuple(items_clicked_last200)} """
            user_detail = pd.read_sql_query(query, engine_rec_db)    
        except:
            items_clicked_sep = []
            for i in items_clicked_last200:
                try: 
                    items_clicked_sep.append(pd.read_sql("SELECT * from item_enc WHERE anbieter_artikelnummer = %s ", engine_rec_db, params = (i, )))
                except:
                    pass
            user_detail = pd.concat(items_clicked_sep).reset_index(drop = True)
        # create return df
        if len(user_detail) > 0:
            anbieterID_enc = user_detail.anbieterid_enc.values.tolist()
            anbietermarktplatz_enc = user_detail.anbietermarktplatz_enc.values.tolist()
            warengruppe_enc = user_detail.warengruppe_enc.values.tolist()
            text_vec = np.array((user_detail.text_vec).values.tolist()[:50]).mean(axis = 0).tolist() ## only use first = last 50 (first = last because it was sorted that way!)
            preis_std = np.array((user_detail.preis_std).values.tolist()).mean(axis = 0)
            minve_std = np.array((user_detail.minve_std).values.tolist()).mean(axis = 0)
            preis_log_std = np.array((user_detail.preis_log_std).values.tolist()).mean(axis = 0)
            minve_log_std = np.array((user_detail.minve_log_std).values.tolist()).mean(axis = 0)
            user = pd.DataFrame({'userid' : str(userid),
                                'usermkt_enc' : mkt_user, 
                                'anbieterid_enc_user' : [anbieterID_enc], 
                                'datum_click' : date,
                                'clicked_before' : [items_clicked_last200],
                                'anbietermarktplatz_enc_user' : [anbietermarktplatz_enc], 
                                'warengruppe_enc_user' : [warengruppe_enc], 
                                'text_vec_user' : [text_vec],
                                'preis_std_user' : preis_std, 
                                'minve_std_user' : minve_std, 
                                'preis_log_std_user': preis_log_std, 
                                'minve_log_std_user': minve_log_std})
        else:
            user = pd.DataFrame({'userid' : str(userid),
                                'usermkt_enc' : mkt_user, 
                                'anbieterid_enc_user' : [[]], 
                                'datum_click' : date,
                                'clicked_before' : [[]],
                                'anbietermarktplatz_enc_user' : [[]], 
                                'warengruppe_enc_user' : [[]], 
                                'text_vec_user' : [[0] * 150],
                                'preis_std_user' : 0, 
                                'minve_std_user' : 0, 
                                'preis_log_std_user': 0, 
                                'minve_log_std_user': 0})
    else: 
        user = pd.DataFrame({'userid' : str(userid),
                                'usermkt_enc' : mkt_user, 
                                'anbieterid_enc_user' : [[]], 
                                'datum_click' : date,
                                'clicked_before' : [[]],
                                'anbietermarktplatz_enc_user' : [[]], 
                                'warengruppe_enc_user' : [[]], 
                                'text_vec_user' : [[0] * 150],
                                'preis_std_user' : 0, 
                                'minve_std_user' : 0, 
                                'preis_log_std_user': 0, 
                                'minve_log_std_user': 0})
    return user

## muss funktion noch laden !!! 
batch_size = 500
for batch in (range(500, len(update_users), batch_size)):
    user_batch = update_users[batch:batch + batch_size].copy()
    user_batch.reset_index (inplace = True, drop = True)
    user_trans_list = []
    for n in range(len(user_batch)):
        user_transformed = get_clicks_aggregate(cursor_maria, engine_rec_db, user_batch.userid[n], user_batch.datum_last_click[n], user_batch.usermkt[n])
        user_trans_list.append(user_transformed)
    # combine all user profiles
    user_trans_df = pd.concat(user_trans_list)
    # Save data to user_enc
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
    update_user_enc_task(conn_rec, users_enc_save_list) #getestet, funktioniert !!!
    print("User batch %s transformed and saved" % (batch, ))
print("all users updated")


get_clicks_aggregate(cursor_maria, engine_rec_db, update_users.userid[0], update_users.datum_last_click[0], update_users.usermkt[0])

q_update_user = """SELECT DISTINCT anbieter_artikelnummer from artikelklicks_current where userID = %s ORDER BY datum DESC LIMIT 200"""
items_clicked = get_user_clicks(cursor_maria, q_update_user, 1569461)
items_clicked
items_clicked_last200 = items_clicked.values.flatten().tolist()
user

try: 
    query = f"""SELECT * from item_enc WHERE anbieter_artikelnummer IN {tuple(items_clicked_last200)} """
    user_detail = pd.read_sql_query(query, engine_rec_db)    
except:
    items_clicked_sep = []
    for i in items_clicked_last200:
        try: 
            items_clicked_sep.append(pd.read_sql("SELECT * from item_enc WHERE anbieter_artikelnummer = %s ", engine_rec_db, params = (i, )))
        except:
            pass
    user_detail = pd.concat(items_clicked_sep).reset_index(drop = True)


user_detail