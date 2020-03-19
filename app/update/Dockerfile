FROM tensorflow/tensorflow:1.15.0-py3

#COPY . /update  

WORKDIR /update

RUN pip install --upgrade pip

RUN pip --quiet install \
    keras \
    numpy \
    pandas \
    bs4 \
    nltk \
    gensim \
    scikit-learn==0.21.3 \
    sqlalchemy==1.3 \
    mysql-connector-python \
    psycopg2-binary 

#CMD ['python', 'update.py']