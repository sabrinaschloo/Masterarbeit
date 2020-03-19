FROM tensorflow/tensorflow:1.15.0-py3

#COPY . /api #remove /app/api everywhere is stupid 

# ich glaube auch wenn ich es nicht reinkopiere kann ich das und unten den Command einstellen !
#WORKDIR /api

RUN pip install --upgrade pip

RUN pip --quiet install \
    keras \
    numpy \
    pandas \
    sqlalchemy==1.3 \
    psycopg2-binary \
    flask \
    scikit-learn==0.21.3 \
    flask_restful \
    waitress 

EXPOSE 2255

CMD ['python', 'api.py']