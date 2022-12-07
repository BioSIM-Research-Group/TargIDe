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
#SVM
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
#Evalutation Metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
from sklearn import metrics

class RM_ESTAGIO_MACHINE_LEARNING:
    def _init_(self):
        pass

    def svm_linear (self,input_file,comb):
        ##WriteS VM_Linear Results in output .csv file
        #os.remove("RESULTS/SVM/svm_linear_10F_27092021.csv")
        file_output=open("RESULTS/SVM/svm_linear_C5_27092021.csv",'a+')
        #file_output.write("\""+str(combs)+"\",")
        #Open input file
        df = pd.read_csv(input_file)
        df.head()
        ##Define features (X) as the list of features determined as most important
        ##.values attribute converts pandas DataFrame into a NumPy array - necessary for distance calculation
        #X = df[['MATS4p','R_TpiPCTPC','cLogS','GATS3m','MDEO-12','HybRatio','GATS4p','MLFER_A','GATS6m','GATS2e']].values
        ##Define features (X) as combinations of n features of the 10 most important ones
        comb_list=list(comb)
        file_output.write(str(comb)+","+"\n")
        X = df[comb_list].values  #.astype(float)
        ##Define Target Value y:
        Y = df['Target ID'].values
        #Dataset division in Train and Test set
        X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.3, random_state=4)
        #TRAIN THE MODEL
        svm = SVC(kernel='linear')
        svm.fit(X_train, Y_train)
        #Predic Answers With Trained Model and Most Accurate K for train and test set
        predict_train =svm.predict(X_train)
        predict_test = svm.predict(X_test)
        #EVALUATE MODEL PREDICTION ABILITY
        ##F1_Score: weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal.
        ###F1_Score for train set
        f1_svm_train = f1_score(Y_train, predict_train, average='weighted')
        ###F1_Score for test set
        f1_svm_test = f1_score(Y_test, predict_test, average='weighted')
        ##Jaccard Similarity Coefficient: size of the intersection divided by the size of the union of two label sets
        ###Jaccard Coefficient for train set
        jaccard_svm_train = jaccard_score(Y_train, predict_train,average='weighted')
        ###Jaccard Coefficient for test set
        jaccard_svm_test = jaccard_score(Y_test, predict_test,average='weighted')
        ##Calculate Accuracy Score:
        ###Accuracy Score for train set
        accuracy_svm_train = accuracy_score(Y_train, predict_train, normalize=True)
        ###Accuracy Score for test set
        accuracy_svm_test = accuracy_score(Y_test, predict_test, normalize=True)
        ##Calculate Precision Score:The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
        #The best value is 1 and the worst value is 0.
        ###Precision score for train set
        precision_svm_train = precision_score(Y_train, predict_train, average='weighted', zero_division=0)
        ###Precision score for test set
        precision_svm_test = precision_score(Y_test, predict_test, average='weighted', zero_division=0)
        ##Calculate Recall Score: recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
        #The best value is 1 and the worst value is 0.
        ###Recall score for train set:
        recall_svm_train = metrics.recall_score(Y_train, predict_train, average='weighted', zero_division=0)
        ###Recall score for test set:
        recall_svm_test = metrics.recall_score(Y_test, predict_test, average='weighted', zero_division=0)
        ##Receiver Operator Characteristic Curve Calculations
        #RocAUC = roc_auc_score(Y_test, predict_test, multi_class='ovr')
        #RocCurve = RocCurveDisplay.from_predictions(Y_test, predict_test)
        #plt.show()
        ##SAVE RESULTS IN OUTPUT .csv FILE
        file_output.write("F1_Score For Train Set"+","+str(f1_svm_train)+"\n")
        file_output.write("F1_Score For Test Set"+","+str(f1_svm_test)+"\n")
        file_output.write("Jaccard_Score For Train Set"+","+str(jaccard_svm_train)+"\n")
        file_output.write("Jaccard_Score For Test Set"+","+str(jaccard_svm_test)+"\n")
        file_output.write("Accuracy For Train Set"+","+str(accuracy_svm_train)+"\n")
        file_output.write("Accuracy For Test Set"+","+str(accuracy_svm_test)+"\n")
        file_output.write("Precision For Train Set"+","+str(precision_svm_train)+"\n")
        file_output.write("Precision For Test Set"+","+str(precision_svm_test)+"\n")
        file_output.write("Recall For Train Set"+","+str(recall_svm_train)+"\n")
        file_output.write("Recall For Test Set"+","+str(recall_svm_test)+"\n")

    def svm_poly (self,input_file,comb):
        ##WriteS VM_Linear Results in output .csv file
        #os.remove("RESULTS/SVM/svm_linear_10F_27092021.csv")
        file_output=open("RESULTS/SVM/svm_poly_C5_27092021.csv",'a+')
        #file_output.write("\""+str(combs)+"\",")
        #Open input file
        df = pd.read_csv(input_file)
        df.head()
        ##Define features (X) as the list of features determined as most important
        ##.values attribute converts pandas DataFrame into a NumPy array - necessary for distance calculation
        #X = df[['MATS4p','R_TpiPCTPC','cLogS','GATS3m','MDEO-12','HybRatio','GATS4p','MLFER_A','GATS6m','GATS2e']].values
        ##Define features (X) as combinations of n features of the 10 most important ones
        comb_list=list(comb)
        file_output.write(str(comb)+","+"\n")
        X = df[comb_list].values  #.astype(float)
        ##Define Target Value y:
        Y = df['Target ID'].values
        #Dataset division in Train and Test set
        X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.3, random_state=4)
        #TRAIN THE MODEL
        svm = SVC(kernel='poly', degree=8)
        svm.fit(X_train, Y_train)
        #Predic Answers With Trained Model and Most Accurate K for train and test set
        predict_train =svm.predict(X_train)
        predict_test = svm.predict(X_test)
        #EVALUATE MODEL PREDICTION ABILITY
        ##F1_Score: weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal.
        ###F1_Score for train set
        f1_svm_train = f1_score(Y_train, predict_train, average='weighted')
        ###F1_Score for test set
        f1_svm_test = f1_score(Y_test, predict_test, average='weighted')
        ##Jaccard Similarity Coefficient: size of the intersection divided by the size of the union of two label sets
        ###Jaccard Coefficient for train set
        jaccard_svm_train = jaccard_score(Y_train, predict_train,average='weighted')
        ###Jaccard Coefficient for test set
        jaccard_svm_test = jaccard_score(Y_test, predict_test,average='weighted')
        ##Calculate Accuracy Score:
        ###Accuracy Score for train set
        accuracy_svm_train = accuracy_score(Y_train, predict_train, normalize=True)
        ###Accuracy Score for test set
        accuracy_svm_test = accuracy_score(Y_test, predict_test, normalize=True)
        ##Calculate Precision Score:The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
        #The best value is 1 and the worst value is 0.
        ###Precision score for train set
        precision_svm_train = precision_score(Y_train, predict_train, average='weighted', zero_division=0)
        ###Precision score for test set
        precision_svm_test = precision_score(Y_test, predict_test, average='weighted', zero_division=0)
        ##Calculate Recall Score: recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
        #The best value is 1 and the worst value is 0.
        ###Recall score for train set:
        recall_svm_train = metrics.recall_score(Y_train, predict_train, average='weighted', zero_division=0)
        ###Recall score for test set:
        recall_svm_test = metrics.recall_score(Y_test, predict_test, average='weighted', zero_division=0)
        ##Receiver Operator Characteristic Curve Calculations
        #RocAUC = roc_auc_score(Y_test, predict_test, multi_class='ovr')
        #RocCurve = RocCurveDisplay.from_predictions(Y_test, predict_test)
        #plt.show()
        ##SAVE RESULTS IN OUTPUT .csv FILE
        file_output.write("F1_Score For Train Set"+","+str(f1_svm_train)+"\n")
        file_output.write("F1_Score For Test Set"+","+str(f1_svm_test)+"\n")
        file_output.write("Jaccard_Score For Train Set"+","+str(jaccard_svm_train)+"\n")
        file_output.write("Jaccard_Score For Test Set"+","+str(jaccard_svm_test)+"\n")
        file_output.write("Accuracy For Train Set"+","+str(accuracy_svm_train)+"\n")
        file_output.write("Accuracy For Test Set"+","+str(accuracy_svm_test)+"\n")
        file_output.write("Precision For Train Set"+","+str(precision_svm_train)+"\n")
        file_output.write("Precision For Test Set"+","+str(precision_svm_test)+"\n")
        file_output.write("Recall For Train Set"+","+str(recall_svm_train)+"\n")
        file_output.write("Recall For Test Set"+","+str(recall_svm_test)+"\n")

    def svm_rbf (self,input_file,comb):
        ##WriteS VM_Linear Results in output .csv file
        #os.remove("RESULTS/SVM/svm_rbf_10F_27092021.csv")
        file_output=open("RESULTS/SVM/svm_rbf_C5_27092021.csv",'a+')
        #file_output.write("\""+str(combs)+"\",")
        #Open input file
        df = pd.read_csv(input_file)
        df.head()
        ##Define features (X) as the list of features determined as most important
        ##.values attribute converts pandas DataFrame into a NumPy array - necessary for distance calculation
        #X = df[['MATS4p','R_TpiPCTPC','cLogS','GATS3m','MDEO-12','HybRatio','GATS4p','MLFER_A','GATS6m','GATS2e']].values
        ##Define features (X) as combinations of n features of the 10 most important ones
        comb_list=list(comb)
        file_output.write(str(comb)+","+"\n")
        X = df[comb_list].values  #.astype(float)
        ##Define Target Value y:
        Y = df['Target ID'].values
        #Dataset division in Train and Test set
        X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.3, random_state=4)
        #TRAIN THE MODEL
        svm = SVC(kernel='rbf')
        svm.fit(X_train, Y_train)
        #Predic Answers With Trained Model and Most Accurate K for train and test set
        predict_train =svm.predict(X_train)
        predict_test = svm.predict(X_test)
        #EVALUATE MODEL PREDICTION ABILITY
        ##F1_Score: weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal.
        ###F1_Score for train set
        f1_svm_train = f1_score(Y_train, predict_train, average='weighted')
        ###F1_Score for test set
        f1_svm_test = f1_score(Y_test, predict_test, average='weighted')
        ##Jaccard Similarity Coefficient: size of the intersection divided by the size of the union of two label sets
        ###Jaccard Coefficient for train set
        jaccard_svm_train = jaccard_score(Y_train, predict_train,average='weighted')
        ###Jaccard Coefficient for test set
        jaccard_svm_test = jaccard_score(Y_test, predict_test,average='weighted')
        ##Calculate Accuracy Score:
        ###Accuracy Score for train set
        accuracy_svm_train = accuracy_score(Y_train, predict_train, normalize=True)
        ###Accuracy Score for test set
        accuracy_svm_test = accuracy_score(Y_test, predict_test, normalize=True)
        ##Calculate Precision Score:The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
        #The best value is 1 and the worst value is 0.
        ###Precision score for train set
        precision_svm_train = precision_score(Y_train, predict_train, average='weighted')
        ###Precision score for test set
        precision_svm_test = precision_score(Y_test, predict_test, average='weighted')
        ##Calculate Recall Score: recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
        #The best value is 1 and the worst value is 0.
        ###Recall score for train set:
        recall_svm_train = metrics.recall_score(Y_train, predict_train, average='weighted', zero_division=0)
        ###Recall score for test set:
        recall_svm_test = metrics.recall_score(Y_test, predict_test, average='weighted', zero_division=0)
        ##Receiver Operator Characteristic Curve Calculations
        #RocAUC = roc_auc_score(Y_test, predict_test, multi_class='ovr')
        #RocCurve = RocCurveDisplay.from_predictions(Y_test, predict_test)
        #plt.show()
        ##SAVE RESULTS IN OUTPUT .csv FILE
        file_output.write("F1_Score For Train Set"+","+str(f1_svm_train)+"\n")
        file_output.write("F1_Score For Test Set"+","+str(f1_svm_test)+"\n")
        file_output.write("Jaccard_Score For Train Set"+","+str(jaccard_svm_train)+"\n")
        file_output.write("Jaccard_Score For Test Set"+","+str(jaccard_svm_test)+"\n")
        file_output.write("Accuracy For Train Set"+","+str(accuracy_svm_train)+"\n")
        file_output.write("Accuracy For Test Set"+","+str(accuracy_svm_test)+"\n")
        file_output.write("Precision For Train Set"+","+str(precision_svm_train)+"\n")
        file_output.write("Precision For Test Set"+","+str(precision_svm_test)+"\n")
        file_output.write("Recall For Train Set"+","+str(recall_svm_train)+"\n")
        file_output.write("Recall For Test Set"+","+str(recall_svm_test)+"\n")

    def svm_sigmoid (self,input_file,comb):
        ##WriteS VM_Linear Results in output .csv file
        #os.remove("RESULTS/SVM/svm_sigmoid_10F_27092021.csv")
        file_output=open("RESULTS/SVM/svm_sigmoid_C5_27092021.csv",'a+')
        #file_output.write("\""+str(combs)+"\",")
        #Open input file
        df = pd.read_csv(input_file)
        df.head()
        ##Define features (X) as the list of features determined as most important
        ##.values attribute converts pandas DataFrame into a NumPy array - necessary for distance calculation
        X = df[['MATS4p','R_TpiPCTPC','cLogS','GATS3m','MDEO-12','HybRatio','GATS4p','MLFER_A','GATS6m','GATS2e']].values
        ##Define features (X) as combinations of n features of the 10 most important ones
        comb_list=list(comb)
        file_output.write(str(comb)+","+"\n")
        X = df[comb_list].values  #.astype(float)
        ##Define Target Value y:
        Y = df['Target ID'].values
        #Dataset division in Train and Test set
        X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.3, random_state=4)
        #TRAIN THE MODEL
        svm = SVC(kernel='sigmoid')
        svm.fit(X_train, Y_train)
        #Predic Answers With Trained Model and Most Accurate K for train and test set
        predict_train =svm.predict(X_train)
        predict_test = svm.predict(X_test)
        #EVALUATE MODEL PREDICTION ABILITY
        ##F1_Score: weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal.
        ###F1_Score for train set
        f1_svm_train = f1_score(Y_train, predict_train, average='weighted')
        ###F1_Score for test set
        f1_svm_test = f1_score(Y_test, predict_test, average='weighted')
        ##Jaccard Similarity Coefficient: size of the intersection divided by the size of the union of two label sets
        ###Jaccard Coefficient for train set
        jaccard_svm_train = jaccard_score(Y_train, predict_train,average='weighted')
        ###Jaccard Coefficient for test set
        jaccard_svm_test = jaccard_score(Y_test, predict_test,average='weighted')
        ##Calculate Accuracy Score:
        ###Accuracy Score for train set
        accuracy_svm_train = accuracy_score(Y_train, predict_train, normalize=True)
        ###Accuracy Score for test set
        accuracy_svm_test = accuracy_score(Y_test, predict_test, normalize=True)
        ##Calculate Precision Score:The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
        #The best value is 1 and the worst value is 0.
        ###Precision score for train set
        precision_svm_train = precision_score(Y_train, predict_train, average='weighted')
        ###Precision score for test set
        precision_svm_test = precision_score(Y_test, predict_test, average='weighted')
        ##Calculate Recall Score: recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
        #The best value is 1 and the worst value is 0.
        ###Recall score for train set:
        recall_svm_train = metrics.recall_score(Y_train, predict_train, average='weighted', zero_division=0)
        ###Recall score for test set:
        recall_svm_test = metrics.recall_score(Y_test, predict_test, average='weighted', zero_division=0)
        ##Receiver Operator Characteristic Curve Calculations
        #RocAUC = roc_auc_score(Y_test, predict_test, multi_class='ovr')
        #RocCurve = RocCurveDisplay.from_predictions(Y_test, predict_test)
        #plt.show()
        ##SAVE RESULTS IN OUTPUT .csv FILE
        file_output.write("F1_Score For Train Set"+","+str(f1_svm_train)+"\n")
        file_output.write("F1_Score For Test Set"+","+str(f1_svm_test)+"\n")
        file_output.write("Jaccard_Score For Train Set"+","+str(jaccard_svm_train)+"\n")
        file_output.write("Jaccard_Score For Test Set"+","+str(jaccard_svm_test)+"\n")
        file_output.write("Accuracy For Train Set"+","+str(accuracy_svm_train)+"\n")
        file_output.write("Accuracy For Test Set"+","+str(accuracy_svm_test)+"\n")
        file_output.write("Precision For Train Set"+","+str(precision_svm_train)+"\n")
        file_output.write("Precision For Test Set"+","+str(precision_svm_test)+"\n")
        file_output.write("Recall For Train Set"+","+str(recall_svm_train)+"\n")
        file_output.write("Recall For Test Set"+","+str(recall_svm_test)+"\n")

    def svm_linear_combinations (self,input_file):
        # This function creates combinations of n 0f the top 10 best features and runs knn function with each combination
        #Open input file
        df = pd.read_csv(input_file)
        df.head()
        ##Define features (X) as combinations of top ten best features
        count_comb=0
        best_features=['MATS4p','R_TpiPCTPC','cLogS','GATS3m','MDEO-12','HybRatio','GATS4p','MLFER_A','GATS6m','GATS2e']
        for column in best_features:
            combs = list(combinations(best_features,4))
        print(combs)
        for comb in combs:
            RM_ESTAGIO_MACHINE_LEARNING.svm_linear("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/DB_BIOFILMS_PA.csv", comb)

    def svm_poly_combinations (self,input_file):
        # This function creates combinations of n 0f the top 10 best features and runs knn function with each combination
        #Open input file
        df = pd.read_csv(input_file)
        df.head()
        ##Define features (X) as combinations of top ten best features
        count_comb=0
        best_features=['MATS4p','R_TpiPCTPC','cLogS','GATS3m','MDEO-12','HybRatio','GATS4p','MLFER_A','GATS6m','GATS2e']
        for column in best_features:
            combs = list(combinations(best_features,4))
        print(combs)
        for comb in combs:
            RM_ESTAGIO_MACHINE_LEARNING.svm_poly("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/DB_BIOFILMS_PA.csv", comb)

    def svm_rbf_combinations (self,input_file):
        # This function creates combinations of n 0f the top 10 best features and runs knn function with each combination
        #Open input file
        df = pd.read_csv(input_file)
        df.head()
        ##Define features (X) as combinations of top ten best features
        count_comb=0
        best_features=['MATS4p','R_TpiPCTPC','cLogS','GATS3m','MDEO-12','HybRatio','GATS4p','MLFER_A','GATS6m','GATS2e']
        for column in best_features:
            combs = list(combinations(best_features,4))
        print(combs)
        for comb in combs:
            RM_ESTAGIO_MACHINE_LEARNING.svm_rbf("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/DB_BIOFILMS_PA.csv", comb)

    def svm_sigmoid_combinations (self,input_file):
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
            RM_ESTAGIO_MACHINE_LEARNING.svm_sigmoid("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/DB_BIOFILMS_PA.csv", comb)

##Function to predict on unknown samples with desired features
##SVM_Predict
    def svm_predict (self,input_file):
        print("SVM PREDICT IS RUNNING")
        #Open input file
        df = pd.read_csv(input_file)
        df.head()
        ##Define features (X) as the list of 4 features with better results
        #X = df[['HybRatio','MDEO-12','GATS6m','MLFER_A']].values
        ##Define features (X) as the list of 5 features with better results
        X = df[['GATS3m','MATS4p','MDEO-12','HybRatio','MLFER_A']].values
        ##Define features (X) as the list of 10 most relevant features 
        #X = df[['MATS4p','R_TpiPCTPC','cLogS','GATS3m','MDEO-12','HybRatio','GATS4p','MLFER_A','GATS6m','GATS2e']].values
        ##Define Target Value y:
        Y = df['Target ID'].values
        #Dataset division in Train and Test set
        X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.3, random_state=4)
        #TRAIN THE MODEL
        svm = SVC(kernel='sigmoid')
        svm.fit(X_train, Y_train)
        #Predic Answers With Trained Model and Most Accurate K for train and test set
        predict_train =svm.predict(X_train)
        predict_test = svm.predict(X_test)
        ##Predict on Unknown sample input file
        test_samples = pd.read_csv("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/UnknowSampleTesting.csv")
        columns_test = test_samples[['GATS3m','MATS4p','MDEO-12','HybRatio','MLFER_A']].values  #.astype(float)
        print(columns_test[0])
        print(columns_test[1])
        print(columns_test[2])
        pred_newsamples1=svm.predict(columns_test[0:14])
        print(pred_newsamples1)



RM_ESTAGIO_MACHINE_LEARNING=RM_ESTAGIO_MACHINE_LEARNING()
#RM_ESTAGIO_MACHINE_LEARNING.svm_linear("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/DB_BIOFILMS_PA.csv")
#RM_ESTAGIO_MACHINE_LEARNING.svm_poly("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/DB_BIOFILMS_PA.csv")
#RM_ESTAGIO_MACHINE_LEARNING.svm_rbf("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/DB_BIOFILMS_PA.csv")
#RM_ESTAGIO_MACHINE_LEARNING.svm_sigmoid("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/DB_BIOFILMS_PA.csv")
#RM_ESTAGIO_MACHINE_LEARNING.svm_linear_combinations("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/DB_BIOFILMS_PA.csv")
#RM_ESTAGIO_MACHINE_LEARNING.svm_poly_combinations("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/DB_BIOFILMS_PA.csv")
#RM_ESTAGIO_MACHINE_LEARNING.svm_rbf_combinations("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/DB_BIOFILMS_PA.csv")
#RM_ESTAGIO_MACHINE_LEARNING.svm_sigmoid_combinations("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/DB_BIOFILMS_PA.csv")
RM_ESTAGIO_MACHINE_LEARNING.svm_predict("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/DB_BIOFILMS_PA.csv")