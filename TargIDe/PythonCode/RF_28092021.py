# Gerar gráficos
import matplotlib.pyplot as plt
# Ler os dados
import pandas as pd
# Visualizar gráficos
import pylab as pl
# Cálculos Matemáticos
import numpy as np
import os
import operator
from math import sqrt
from itertools import combinations
# Random Forest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#Evalutation Metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score

class RM_ESTAGIO_MACHINE_LEARNING:
    def _init_(self):
        pass

#Random Forest (RF)
#Function random_forest uses RF classification model
#Input is Database.csv, X are columns to be used as features and Y is target column (the response)
    def random_forest (self,input_file):
        ##Write NN Results in output .csv file
        #os.remove("RESULTS/RF/RF_F10_27092021.csv")
        file_output=open("RESULTS/RF/RF_F10_27092021.csv",'a+')
        #file_output.write("\""+str(combs)+"\",")
        #Open input file
        df = pd.read_csv(input_file)
        df.head()
        ##Define features (X) as the list of features determined as most important
        ##.values attribute converts pandas DataFrame into a NumPy array - necessary for distance calculation
        X = df[['MATS4p','R_TpiPCTPC','cLogS','GATS3m','MDEO-12','HybRatio','GATS4p','MLFER_A','GATS6m','GATS2e']].values
        ##Define features (X) as combinations of n features of the 10 most important ones
        #comb_list=list(comb)
        #file_output.write(str(comb)+","+"\n")
        #X = df[comb_list].values  #.astype(float)
        ##Define Target Value y:
        Y = df['Target ID'].values
        #Dataset division in Train and Test set
        X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.3, random_state=4)
        #TRAIN THE MODEL
        RF = RandomForestClassifier(n_estimators=100)
        RF = RF.fit(X_train,Y_train)
        #Predic Answers With Trained Model and Most Accurate K for train and test set
        predict_train = RF.predict(X_train)
        predict_test = RF.predict(X_test)
        #EVALUATE MODEL PREDICTION ABILITY
        ##F1_Score: weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal.
        ###F1_Score for train set
        f1_rf_train = f1_score(Y_train, predict_train, average='weighted')
        ###F1_Score for test set
        f1_rf_test = f1_score(Y_test, predict_test, average='weighted')
        ##Jaccard Similarity Coefficient: size of the intersection divided by the size of the union of two label sets
        ###Jaccard Coefficient for train set
        jaccard_rf_train = jaccard_score(Y_train, predict_train,average='weighted')
        ###Jaccard Coefficient for test set
        jaccard_rf_test = jaccard_score(Y_test, predict_test,average='weighted')
        ##Calculate Accuracy Score:
        ###Accuracy Score for train set
        accuracy_rf_train = accuracy_score(Y_train, predict_train, normalize=True)
        ###Accuracy Score for test set
        accuracy_rf_test = accuracy_score(Y_test, predict_test, normalize=True)
        ##Calculate Precision Score:The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
        #The best value is 1 and the worst value is 0.
        ###Precision score for train set
        precision_rf_train = precision_score(Y_train, predict_train, average='weighted', zero_division=0)
        ###Precision score for test set
        precision_rf_test = precision_score(Y_test, predict_test, average='weighted', zero_division=0)
        ##Calculate Recall Score: recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
        #The best value is 1 and the worst value is 0.
        ###Recall score for train set:
        recall_rf_train = recall_score(Y_train, predict_train, average='weighted', zero_division=0)
        ###Recall score for test set:
        recall_rf_test = recall_score(Y_test, predict_test, average='weighted', zero_division=0)
        ##Receiver Operator Characteristic Curve Calculations
        #RocAUC = roc_auc_score(Y_test, predict_test, multi_class='ovr')
        #RocCurve = RocCurveDisplay.from_predictions(Y_test, predict_test)
        #plt.show()
        ##SAVE RESULTS IN OUTPUT .csv FILE
        file_output.write("F1_Score For Train Set"+","+str(f1_rf_train)+"\n")
        file_output.write("F1_Score For Test Set"+","+str(f1_rf_test)+"\n")
        file_output.write("Jaccard_Score For Train Set"+","+str(jaccard_rf_train)+"\n")
        file_output.write("Jaccard_Score For Test Set"+","+str(jaccard_rf_test)+"\n")
        file_output.write("Accuracy For Train Set"+","+str(accuracy_rf_train)+"\n")
        file_output.write("Accuracy For Test Set"+","+str(accuracy_rf_test)+"\n")
        file_output.write("Precision For Train Set"+","+str(precision_rf_train)+"\n")
        file_output.write("Precision For Test Set"+","+str(precision_rf_test)+"\n")
        file_output.write("Recall For Train Set"+","+str(recall_rf_train)+"\n")
        file_output.write("Recall For Test Set"+","+str(recall_rf_test)+"\n")

#RANDOM FOREST WITH UNKNOWN SAMPLES FOR TESTING
    def random_forest_predict (self,input_file):     
        print("RUNNING RANDOM FOREST FOR PREDICTION")
        #Open input file
        df = pd.read_csv(input_file)
        df.head()
        ##Define features (X) as the 4 best features
        X = df[['MATS4p','cLogS','GATS6m','GATS2e']].values
        ##Define features (X) as the 5 best features
        #X = df[['MDEO-12','cLogS','HybRatio','MLFER_A','GATS2e']].values
        ##Define features (X) as the list of 10 most relevant features 
        #X = df[['MATS4p','R_TpiPCTPC','cLogS','GATS3m','MDEO-12','HybRatio','GATS4p','MLFER_A','GATS6m','GATS2e']].values
        ##Define Target Value y:
        Y = df['Target ID'].values
        #Dataset division in Train and Test set
        X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.3, random_state=4)
        #TRAIN THE MODEL
        RF = RandomForestClassifier(n_estimators=100)
        RF = RF.fit(X_train,Y_train)
        #Predic Answers With Trained Model and Most Accurate K for train and test set
        predict_train = RF.predict(X_train)
        predict_test = RF.predict(X_test)
        ##Predict on Unknown sample input file
        test_samples = pd.read_csv("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/UnknowSampleTesting.csv")
        columns_test = test_samples[['MATS4p','cLogS','GATS6m','GATS2e']].values  #.astype(float)
        print(columns_test[0])
        print(columns_test[1])
        print(columns_test[2])
        pred_newsamples=RF.predict(columns_test[0:14])
        print(pred_newsamples)
        #Predict on aBiofilm Samples
        aBiofilm = pd.read_csv("/Users/Rita Magalhaes/Desktop/Estágio JC/aBiofilm/aBiofilmAgents.csv")
        columns_abiofilm = aBiofilm[['MATS4p','cLogS','GATS6m','GATS2e']].values  #.astype(float)
        pred_abiofilm = RF.predict(columns_abiofilm[0:27])
        print(pred_abiofilm)

    def random_forest_combinations (self,input_file):
        # This function creates combinations of n 0f the top 10 best features and runs knn function with each combination
        #Open input file
        df = pd.read_csv(input_file)
        df.head()
        ##Define features (X) as combinations of top ten best features
        count_comb=0
        best_features=['MATS4p','R_TpiPCTPC','cLogS','GATS3m','MDEO-12','HybRatio','GATS4p','MLFER_A','GATS6m','GATS2e']
        for column in best_features:
            combs = list(combinations(best_features,5))
        print(combs)
        for comb in combs:
            RM_ESTAGIO_MACHINE_LEARNING.random_forest("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/DB_BIOFILMS_PA.csv", comb)

RM_ESTAGIO_MACHINE_LEARNING=RM_ESTAGIO_MACHINE_LEARNING()
#RM_ESTAGIO_MACHINE_LEARNING.random_forest("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/DB_BIOFILMS_PA.csv")
#RM_ESTAGIO_MACHINE_LEARNING.random_forest_combinations("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/DB_BIOFILMS_PA.csv")
RM_ESTAGIO_MACHINE_LEARNING.random_forest_predict("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/DB_BIOFILMS_PA.csv")