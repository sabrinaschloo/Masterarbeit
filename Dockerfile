FROM jupyter/tensorflow-notebook:2ce7c06a61a1

RUN pip install --upgrade pip

RUN pip --quiet install \
    keras \
    gensim \
    scikit-learn==0.21.3 \
    nltk \
    feather-format \
    bs4 \
    hyperopt \
    hyperas \
    scikit-plot