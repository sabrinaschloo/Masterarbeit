# Masterarbeit
A Recommender System for Retail Sourcing based on Deep Learning

This repository includes all code for the development and evaluation of the Hybrid NeuMF developed for the wholesale marketplace zentrada.

## Model Development
The code can be run with the docker image sabrinaschloo/dl_recommender:notebook, it inlcudes all necessary packages and should be started with the command:

  docker run -8888:8888 -v "location_of_masterarbeit":/home/jovyan --name tensorflow-notebook sabrinaschloo/dl_recommender:notebook

All necessary data is in the google drive folder "data", which needs to be copied into this repository.

The notebooks describe the following steps in development:
- 01-03 describe the analysis and preprocessing of training data 
- 04 show the training of the Hybrid NeuMF model and alternatives with different regularization, 04_01 is the final model
- 05 evaluate the different training setups, 05_01 is a more detailed analysis of the final model
- 06 extract embedding vectors for visualization with the embedding projector
- 07-08 train and evaluate the baseline models NeuMF and Hybrid NeuMF Text

## Model Implementation
The implementation of the final model can be found in /app
- /api is the code implemented for the API to query the model. The API can be tested with the docker image sabrinaschloo/dl_recommender:api, which contains the necessary packages, scripts, and model. The container makes the API available on port 2255, it should be started with the following command:
  
  docker run -p 2255:2255 -v "location_of_data":/data --name=person_api -d sabrinaschloo/dl_recommeder:api
  
The API can be queried with the example code in /app/query_api.py or an example can be imported to postman from  https://www.getpostman.com/collections/94e447a37e159325eccd. 
 
- /update is the code implemented for the update container, which keeps the item and user profile up to date. This code cannot be run, since the connection to the live data base is not available. 
