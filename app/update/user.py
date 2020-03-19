import pickle
import pandas as pd
import numpy as np


# pre = "app/update/" # auch in funktionen ändern !
pre = "/update/"

# load encoder
label_enc_mkt = pickle.load( open( pre +"preprocessing/label_mkt.pkl", "rb" ) )

def get_new_users (cursor, max_date):
    query = (" SELECT userID, erstRegMarktplatz, registrierungsdatum from userstammdaten where registrierungsdatum > %s")
    cursor.execute(query, (str(max_date),))
    data = cursor.fetchall()
    data = pd.DataFrame(data)
    data.columns = ['userid', 'usermkt', 'regdate']
    return data


def encode_mkt (usermkt):
    usermkt_enc = (label_enc_mkt.transform(usermkt.fillna("EU").values) + 1 ).tolist()
    return usermkt_enc

def fill_empty(df):
    all_new_list = []
    for n in range(len(df)):
        user = df[n: n+1]
        new_users_info = pd.DataFrame({'userid' : user.userid,
            'datum_click' : user.regdate,
            'clicked_before' : [[]],
            'anbieterid_enc_user' : [[]], 
            'anbietermarktplatz_enc_user' : [[]], 
            'warengruppe_enc_user' : [[]], 
            'text_vec_user' : [[0] * 150],
            'preis_std_user' : 0, 
            'minve_std_user' : 0, 
            'preis_log_std_user': 0, 
            'minve_log_std_user' : 0})
        df_user = pd.merge(user, new_users_info, how = "left", on = "userid")
        all_new_list.append(df_user) 
    final = pd.concat(all_new_list, sort = True)
    return final


def insert_user_initial_task(conn, task):
    """
    update all parameters in item_enc
    :param conn:
    :param task:
    """
    sql = ''' INSERT INTO user_enc (anbieterid_enc_user, datum_click, clicked_before, anbietermarktplatz_enc_user, warengruppe_enc_user, text_vec_user,  preis_std_user, minve_std_user, preis_log_std_user, minve_log_std_user, usermkt_enc, userid) 
                VALUES (%s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s) 
                ON CONFLICT (userid)
                DO NOTHING '''
    
    cur = conn.cursor()
    cur.executemany(sql, task) # task is list of mulitple updates
    conn.commit()


def get_user_clicks (cursor_maria, query, userid):
    cursor_maria.execute(query, (int(userid),))
    data = cursor_maria.fetchall()
    data = pd.DataFrame(data)
    return data

def get_clicks_aggregate (cursor_maria, engine_rec_db, userid, date, usermkt): 
    # transform usermkt ##############
    mkt_user = (label_enc_mkt.transform(np.array(usermkt).reshape(1)) + 1 )[0]
    # get last 200 clicks 
    q_update_user = """SELECT DISTINCT anbieter_artikelnummer from artikelklicks_current where userID = %s ORDER BY datum DESC LIMIT 200"""
    items_clicked = get_user_clicks(cursor_maria, q_update_user, userid)
    items_clicked_last200 = items_clicked[0].values.tolist()
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
    return user


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