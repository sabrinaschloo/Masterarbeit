from waitress import serve
from flask import Flask, request
from flask_restful import Resource, Api 
import tensorflow as tf

import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
import pickle 

# import sqlalchemy as db
# from sqlalchemy import create_engine
# import psycopg2
import sqlite3

import recommend as recommend
import NeuMF as NeuMF

# DB connection
#DATABASE_URI = 'postgres+psycopg2://postgres:power2020@host.docker.internal:5431/person_recommender' # needs change
#DATABASE_URI = 'postgres+psycopg2://postgres:power2020@co1.db.schimmel:5432/person_recommender' # needs change
#engine = create_engine(DATABASE_URI)
engine = sqlite3.connect('/data/db.db')


app = Flask(__name__)
#app.config["DEBUG"] = True
api = Api (app)
model = None

def load_model():
    # load the pre-trained Keras model (here we are using a model
    global model
    model = NeuMF.get_model(layers = [128, 64, 32, 16], reg_layers=[0.001, 0.001, 0.001, 0.001], reg_mf=0.001)
    # load weights into new model
    model.load_weights("/api/model/weights.h5")
    model._make_predict_function()


class homepage(Resource):
   def get(self):
        # get parameters
        userid = int(request.args.get('userID'))
        # create df of input data in body
        json = request.get_json(force = True)
        listing = []
        for i in ['bestseller', 'rabatt', 'neu']:
            data = json_normalize(data = json [i])
            data.loc[:,'position'] = i
            listing.append(data)
        df_input = pd.concat(listing) 
        df_input.drop('score', axis = 1, inplace = True)
        global sess
        global graph
        with graph.as_default():
            tf.compat.v1.keras.backend.set_session(sess)
            result = recommend.rank_list(model, engine, userid, df_input) # later remove item_enc & user_enc !!
        return (result)



api.add_resource(homepage, '/personalized/homepage')

# load model and start api
#tf_config = some_custom_config
#sess = tf.Session(config=tf_config)
sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()

tf.compat.v1.keras.backend.set_session(sess)
load_model()

serve (app, host = "0.0.0.0", port = 2255)
