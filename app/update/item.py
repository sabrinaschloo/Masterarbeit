import pickle
import gensim
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import nltk
#nltk.download('wordnet')
nltk.data.path.append("nltk_data")
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

# pre = "app/update/" # auch in funktionen ändern ! 
pre = "/update/"

# load all transformers
# cat
label_enc_id = pickle.load( open( pre + "preprocessing/label_anbieterID.pkl", "rb" ) )
label_enc_mkt = pickle.load( open( pre +"preprocessing/label_mkt.pkl", "rb" ) )
label_enc_wg = pickle.load( open( pre +"preprocessing/label_warengruppe.pkl", "rb" ) )

# cont impute
imputer_stueck = pickle.load( open( pre +"preprocessing/imputer_stueck.pkl", "rb" ) )
imputer_preis = pickle.load( open( pre +"preprocessing/imputer_preis.pkl", "rb" ) )
# cont std old
std_stueck = pickle.load( open( pre +"preprocessing/scaler_stueck.pkl", "rb" ) )
std_preis = pickle.load( open( pre +"preprocessing/scaler_preis.pkl", "rb" ) )
# cont std new
std_stueck_log = pickle.load( open( pre +"preprocessing/new_scaler_ve_log.pkl", "rb" ) )
std_preis_log = pickle.load( open( pre +"preprocessing/new_scaler_preis_log.pkl", "rb" ) )

# text preprocessing and embedding
my_stop_words = STOPWORDS.union(set(['size', 'color', 'material', 'product', 'dimension', 'length', 'package', 'brand', 
                     'pack', 'width', 'piece', 'height', 'quality', 'group', 'high', 'model', 'article', 'assort',
                     'price', 'weight', 'colour', 'products', 'type', 'design', 'diameter']))

model = Word2Vec.load(pre + "preprocessing/word2vec.model")
word_vectors = model.wv


def get_new_clicks (cursor, max_date):
    query = """SELECT cl.datum, cl.userID, cl.artikelID, cl.anbieterID, cl.anbieter_artikelnummer,
        u.erstRegMarktplatz, 
        a.anbietermarktplatz,
        i.erstanlageDatum, i.waehrung, i.stueckpreis, i.stueck_pro_ve,
        w.warengruppe,
        t.bezeichnung, t.beschreibung
        FROM artikelklicks_current as cl
        LEFT JOIN userstammdaten as u
        on cl.userID = u.userID
        LEFT JOIN anbieterstammdaten as a
        ON cl.anbieterID = a.anbieterID
        LEFT JOIN artikelstammdaten as i
        ON cl.artikelID = i.artikelID
        LEFT JOIN artikelwarengruppe as w
        ON cl.artikelID = w.artikelID
        LEFT JOIN artikeltexte as t
        ON cl.artikelnummer = t.artikelnummer AND cl.anbieterID = t.firmaID
        WHERE (t.marktplatz = 'EN' or t.marktplatz = 'EU') and cl.datum > %s """
    cursor.execute(query, (str(max_date),))
    data = cursor.fetchall()
    data = pd.DataFrame(data)
    data.columns = ['datum_last_click', 'userid', 'artikelid', 'anbieterid', 'anbieter_artikelnummer', 'usermkt', 'anbietermarktplatz', 
                    'erstanlagedatum', 'waehrung', 'stueckpreis', 'stueck_pro_ve', 'warengruppe', 'bezeichnung',
                    'beschreibung']
    return data

def price_to_euro(df1):
    df = df1.copy()
    df.loc[:,'stueckpreis'] = df['stueckpreis'].astype(float)
    df.loc[:,'exchangeRate'] = np.where(df['waehrung'] == 'EUR', float(1) , 0)
    df.loc[:,'exchangeRate'] = np.where(df['waehrung'] == 'PLN', float(0.228786) , df['exchangeRate'] )
    df.loc[:,'exchangeRate'] = np.where(df['waehrung'] == 'HUF', float(0.00299247), df['exchangeRate'] )
    df.loc[:,'preis_euro'] = df['stueckpreis'] * df['exchangeRate']
    return df['preis_euro']


## Transform Data
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in my_stop_words and len(token) > 3: # standard stopwords and remove shorter 3 
            result.append(WordNetLemmatizer().lemmatize(token, pos='v'))
    return result


# get average vector of words in description
def get_mean_vector(w2v_model, word_vecs, words):
    # remove out-of-vocabulary words
    words = [word for word in words if word in word_vecs.vocab] 
    if len(words) >= 1:
        return np.mean(w2v_model[words], axis=0) # average the vecs
    else:
        return []

# transform all data
def transform_data(df):
    # fill empty warengruppe
    df1 = pd.DataFrame({'anbieter_artikelnummer': df.anbieter_artikelnummer})
    # impute preis and stueck_ve
    df1.loc[:,"minve_imp"] = imputer_stueck.transform(df[['stueck_pro_ve']])
    df1.loc[:,"preis_imp"] = imputer_preis.transform(df[['preis_euro']])
    # scale preis and stueck_ve
    df1.loc[:,"minve_std"] = std_stueck.transform(df1[['minve_imp']])
    df1.loc[:,"preis_std"] = std_preis.transform(df1[['preis_imp']])
    # log and scale preis and stueck_ve
    df1.loc[:,"minve_log"] = np.log(df1.minve_imp+1e-2)
    df1.loc[:,"preis_log"] = np.log(df1.preis_imp+1e-2)
    df1.loc[:,"minve_log_std"] = std_stueck_log.transform(df1[['minve_log']])
    df1.loc[:,"preis_log_std"] = std_preis_log.transform(df1[['preis_log']])
    df1.drop (['minve_imp', 'minve_log', 'preis_imp', 'preis_log'], axis = 1, inplace = True)
    # Create label-encodings anbieterID, anbietermarktplatz, warengruppe (need to add +1 to each label and insert 0 for unknown labels)
    anbieterID = []
    for n in range(len(df)):
            try:
                anbieterID.append(label_enc_id.transform(df[n:n+1][['anbieterid']])[0]+1)
            except: 
                anbieterID.append(0)
     # need to fill the items with currently no wg with NN 
    warengruppe = []
    for n in range(len(df)):
            try:
                warengruppe.append(label_enc_wg.transform(df[n:n+1]['warengruppe'])[0]+1)
            except: 
                warengruppe.append(0)
    df1.loc[:,'anbietermarktplatz_enc'] = (label_enc_mkt.transform(df.anbietermarktplatz.values) + 1 ).tolist()
    # Transform text 
    text_vec_list = []
    # Change text_vec to list
    for n in range(len(df)):
            try:
                prepro = preprocess(df.text_clean[n])
                ls = get_mean_vector(model, word_vectors, prepro).tolist()
                text_vec_list.append(ls)
            except: 
                ls = [0] * 150
                text_vec_list.append(ls)
    # combine data to second df
    df2 = pd.DataFrame({"anbieter_artikelnummer": df.anbieter_artikelnummer,
                        "erstanlagedatum": df.erstanlagedatum,
                        'text_vec': text_vec_list, 
                        "anbieterid_enc" : anbieterID, 
                        "warengruppe_enc": warengruppe})
    df = pd.merge(df1, df2, how = "left", on = "anbieter_artikelnummer")
    return(df)



## Save data
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