
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
# Pickle
import pickle
#Datetime
from datetime import datetime
# Importar linear regression model
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
#Evalutation of Logistic regression 
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from itertools import combinations
from sklearn.metrics import log_loss
#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
#NN
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.datasets import make_classification
#SVM
from sklearn import svm
from sklearn.metrics import f1_score

Best_comb_svm={}

class RM_ESTAGIO_MACHINE_LEARNING:
    def _init_(self):
        pass

# Criar a função para ler base de dados
    def ler_base_dados_biofilm (self,input_file):
        df = pd.read_csv(input_file)
        # Ler os títulos das colunas
        heads=df.head()
        #print(heads)
        # Resumo das estatísticas
        Resume_stat=df.describe()
        #print (Resume_stat)
        return df
    
# Linear regression
    def linear_regression_biofilm (self,input_file):
        df = pd.read_csv(input_file)
        param_test1 = df[['Total Molweight','cLogP','Total Surface Area']]
        visualize1 = param_test1
        visualize1.hist()
        plt.show()
        msk = np.random.rand(len(df)) < 0.8
        train = param_test1[msk]
        test = param_test1[~msk]
        regr = linear_model.LinearRegression()
        train_x = np.asanyarray(train[['Total Molweight']])
        train_y = np.asanyarray(train[['cLogP']])
        regr.fit (train_x, train_y)
        #The coefficients
        print ('Coefficients: ', regr.coef_)
        print ('Intercept: ',regr.intercept_)
        #Avaliar o modelo de LR
        test_x = np.asanyarray(test[['Total Molweight']])
        test_y = np.asanyarray(test[['cLogP']])
        test_y_ = regr.predict(test_x)
        print("Erro médio absoluto: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
        print("Soma residual dos quadrados (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
        print("R2-score: %.2f" % r2_score(test_y_ , test_y) )
        
    #Logistic Regression
    def logistic_regression_biofilm (self,input_file):
        df = pd.read_csv(input_file)
        #df=df.reset_index()
        #df.fillna(0)
        param_test1 = df[['Total Molweight','cLogP','Total Surface Area','fragC'  ]]
        X = np.asarray(param_test1)
        X[0:5]
        Y = np.asarray(df['Target ID'])
        Y[0:5]
        X = preprocessing.StandardScaler().fit(X).transform(X)
        X[0:5]
        X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=4)
        print ('Train set:', X_train.shape,  Y_train.shape)
        print ('Test set:', X_test.shape,  Y_test.shape)
        LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,Y_train)
        LR
        yhat = LR.predict(X_test)
        yhat
        #print(yhat)
        yhat_prob = LR.predict_proba(X_test)
        yhat_prob
        #print(yhat_prob)
        # EVALUATE LOGISTIC REGRESSION
        jaccard_score(Y_test, yhat, average='macro')
        # Log Loss
        log_loss(Y_test, yhat_prob)

    # Confusion Matrix
    def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')
            print(cm)
            
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            print(confusion_matrix(Y_test, yhat, labels=[1,0]))
            # Compute confusion matrix
            cnf_matrix = confusion_matrix(Y_test, yhat, labels=[1,0])
            np.set_printoptions(precision=2)
            # Plot non-normalized confusion matrix
            plt.figure()
            plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
            print (classification_report(Y_test, yhat))

#KNN
#Function k_nearest_neighbors uses KNN classification model
#Input is Database.csv, X are columns to be used as features and Y is target column (the response)
    def k_nearest_neighbors (self,input_file,comb):
        ##Write KNN Results in output .csv file
        #file_output=open("RESULTS/Results_25_75_4.csv",'a+')
        #file_output.write("\""+str(list_of_cols)+"\",")
        #Open input file
        df = pd.read_csv(input_file)
        df.head()
        #file_output=open("KNN_25_75_4.csv",'w+')
        comb_list=list(comb)
        #print (comb_list)
        ##Define features (X) as the list of combinations defined in function select_best_columns and called through function knn_best_features:
        X = df[comb_list].values  #.astype(float)
        ##Define Target Values:
        Y = df['Target ID'].values
        ##Normalize the Data:
        X = preprocessing.StandardScaler().fit(X).transform(X)
        #print(X)
        #Dataset division in Train and Test set
        X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=4)
        #print ('Training set:', X_train.shape,  Y_train.shape)
        #print ('Test set:', X_test.shape,  Y_test.shape)
        #Test Model With Different Ks from 1 to Ks
        Ks = 15
        mean_acc = np.zeros((Ks-1))
        std_acc = np.zeros((Ks-1))
        for n in range(1,Ks):
            #Train Model and Predict  
            knn_model = KNeighborsClassifier(n_neighbors = n).fit(X_train,Y_train)
            yhat=knn_model.predict(X_test)
            mean_acc[n-1] = metrics.accuracy_score(Y_test, yhat)
            std_acc[n-1]=np.std(yhat==Y_test)/np.sqrt(yhat.shape[0])
            Train_Set_Certainty=metrics.accuracy_score(Y_train, knn_model.predict(X_train))
            Test_Set_Certainty=metrics.accuracy_score(Y_test, yhat)
            #print(n)
            #print(Train_Set_Certainty)
            #print(Test_Set_Certainty)
        #Plot Model Accuracy with different Ks 
        #plt.plot(range(1,Ks),mean_acc,'g')
        #plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
        #plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
        #plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
        #plt.ylabel('Accuracy ')
        #plt.xlabel('Number of Neighbors (K)')
        #plt.tight_layout()
        #plt.show()
        #Best K Value
        #print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)
        #Train the model with best K:
        mean_acc
        k = mean_acc.argmax()+1
        knn_model_bestaccuracy = KNeighborsClassifier(n_neighbors = k).fit(X_train,Y_train)
        #Predic Answers With Trained Model and Most Accurate K
        yhat = knn_model_bestaccuracy.predict(X_test)
        #Compare Predicted Answers with Real Answers / Evaluate the KNN Model:
        #Train_Set_Certainty=metrics.accuracy_score(Y_train, knn_model_bestaccuracy.predict(X_train))
        #Test_Set_Certainty=metrics.accuracy_score(Y_test, yhat)
        #Calculate F1_Score:
        f1_knn= f1_score(Y_test, yhat, average='weighted')
        print(f1_knn)
        #Calculate Jaccard Score:
        #Jaccard score is related with accuracy
        jaccard_knn = jaccard_score(Y_test,yhat,average='weighted')
        #print(jaccard_poly)
        #Calculate Accuracy Score:
        certainty_knn = metrics.accuracy_score(Y_test, yhat)
        ##Sort results from highest to lowest according to F1_Score, saving all three scoring metrics until the number mentioned in first line:
        if len(Best_comb_svm)<210:
            Best_comb_svm[comb]=[f1_knn,jaccard_knn,certainty_knn]
        if len(Best_comb_svm)==210:
            sorted_comb_dict = sorted(Best_comb_svm.items(), key=operator.itemgetter(1),reverse=False)
            lowest_comb=sorted_comb_dict[209][0]
            lowest_f1_value=Best_comb_svm[lowest_comb][0]
            print (lowest_f1_value)
            if f1_knn>lowest_f1_value:
                del Best_comb_svm[lowest_comb]
                Best_comb_svm[comb]=[f1_knn,jaccard_knn,certainty_knn]
            print (sorted_comb_dict[0:2])
        #print(Train_Set_Certainty)
        #print(Test_Set_Certainty)
        #file_output.write(str(mean_acc.max())+","+ str(mean_acc.argmax()+1)+",")
        #file_output.write(str(Train_Set_Certainty)+",")
        #file_output.write(str(Test_Set_Certainty)+","+"KNN"+"\n")
        #Test Model For Unknown Sample, input is .csv with unknown ligands
        #read_test_samples = pd.read_csv("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/UnknowSampleTesting.csv")
        #A = read_test_samples[['Total Molweight','cLogP','Total Surface Area', 'Polar Surface Area','fragC']].values  #.astype(float)
        #print(A[0])
        #print(A[1])
        #print(A[2])
        #pred=knn_model_bestaccuracy.predict(A[0:12])
        #print(pred)
        #print(X_test)

#Neural Networks
#Function neural networks uses Neural Networks 
#Input is Database.csv, X are columns to be used as features and Y is target column (the response)
    def neural_networks (self,input_file,comb):
        df = pd.read_csv(input_file)
        df.head()
        comb_list=list(comb)
        #print (comb_list)
        ##Define features (X) as the list of combinations defined in function select_best_columns and called through function nn_best_features:
        X = df[comb_list].values  #.astype(float)
        ##Define Target Values:
        Y = df['Target ID'].values
        #Define Train and Test Sets
        X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.30, random_state=40)
        #print(X_train.shape); print(X_test.shape)
        #print(X_train[0:5])
        #print(X_test[0:5])
        #Train Model:
        mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=4000)
        mlp.fit(X_train,Y_train)
        #Predict results for training and testing set:
        predict_train = mlp.predict(X_train)
        predict_test = mlp.predict(X_test)
        #Evaluate Model on Train and Test Set:
        #print(confusion_matrix(Y_train,predict_train))
        #print(classification_report(Y_train,predict_train))
        #print(confusion_matrix(Y_test,predict_test))
        #print(classification_report(Y_test,predict_test))
        #Evaluate NN
        #F1 Score:
        f1_nn = f1_score(Y_test, predict_test, average='weighted')
        #print(f1_nn)
        #Jaccard Score:
        jaccard_nn = jaccard_score(Y_test, predict_test,average='weighted')
        #Accuracy Score:
        certainty_nn = metrics.accuracy_score(Y_test, predict_test)
        #Sort Best Results according to f1_Score up to the number of scores mentioned in first line:
        if len(Best_comb_svm)<50:
            Best_comb_svm[comb]=[f1_nn,jaccard_nn,certainty_nn]
        if len(Best_comb_svm)==50:
            sorted_comb_dict = sorted(Best_comb_svm.items(), key=operator.itemgetter(1),reverse=False)
            lowest_comb=sorted_comb_dict[49][0]
            lowest_f1_value=Best_comb_svm[lowest_comb][0]
            #print (lowest_f1_value)
            if f1_nn>lowest_f1_value:
                del Best_comb_svm[lowest_comb]
                Best_comb_svm[comb]=[f1_nn,jaccard_nn,certainty_nn]
            #print (sorted_comb_dict[0:2])
        #Test Model with Unknown Samples
        #read_test_samples = pd.read_csv("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/UnknowSampleTesting.csv")
        #A = read_test_samples[['Total Molweight','cLogP','Total Surface Area', 'Polar Surface Area','fragC']].values  #.astype(float)
        #print(A[0:12])
        #predict_unknown = mlp.predict(A[0:5])
        #print(predict_unknown)

#Support Vector Machines
#Function svm uses Support Vector Machines, input is .csv or .pickle file with features/combinations
    def svm (self,input_file,comb):
        df = pd.read_csv(input_file)
        df.head()
        #X = df[['Total Molweight','cLogP','Total Surface Area']].values  #.astype(float)
        comb_list=list(comb)
        #print (comb_list)
        ##Define features (X) as the list of combinations defined in function select_best_columns and called through function svm_best_features:
        X = df[comb_list].values  #.astype(float)
        ##Define Target Values
        Y = df['Target ID'].values
        ##Split into Training and Testing set:
        X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=4)
        #print ('Training set:', X_train.shape,  Y_train.shape)
        #print ('Test set:', X_test.shape,  Y_test.shape)
        ##DIfferent way of defining features and target, from model documentation, do not use:
        #X, Y = make_classification(n_samples=10, random_state=0)
        ##SVM WITH LINEAR KERNEL
        #linear_svm = svm.SVC(kernel='linear')
        #print(linear_svm)
        # linear kernel computation
        #gram_train_linear = np.dot(X_train, X_train.T)
        #linear_svm.fit(gram_train_linear, Y_train)
        # predict on training examples
        #gram_test_linear = np.dot(X_test, X_train.T)
        #predict_linear = linear_svm.predict(gram_test_linear)
        #print(predict_linear)
        #Evaluate Linear
        #f1_linear = f1_score(Y_test, predict_linear, average='weighted')
        #print(f1_linear)
        #Jaccard score is related with accuracy
        #jaccard_linear = jaccard_score(Y_test, predict_linear,average='weighted')
        #print(jaccard_linear)
        #Accuracy Linear
        #certainty_linear = metrics.accuracy_score(Y_test, predict_linear)
        #print(certainty_sigmoidal)
        #Sort Best Linear SVM Results according to F1_score 
        #if len(Best_comb_svm)<210:
         #   Best_comb_svm[comb]=[jaccard_linear,f1_linear,certainty_linear]
        #if len(Best_comb_svm)==210:
            #sorted_comb_dict = sorted(Best_comb_svm.items(), key=operator.itemgetter(1),reverse=False)
            #lowest_comb=sorted_comb_dict[209][0]
            #lowest_jaccard_value=Best_comb_svm[lowest_comb][0]
            #print (lowest_f1_value)
            #if jaccard_linear>lowest_jaccard_value:
                #del Best_comb_svm[lowest_comb]
                #Best_comb_svm[comb]=[jaccard_linear,f1_linear,certainty_linear]
        ##SVM WITH Radial Basis Function (RBF)
        #rbf_svc = svm.SVC(kernel='rbf')
        #print(rbf_svc)
        # rbf kernel computation
        #gram_train_rbf = np.dot(X_train, X_train.T)
        #rbf_svc.fit(gram_train_rbf, Y_train)
        # predict on training examples
        #gram_test_rbf = np.dot(X_test, X_train.T)
        #predict_rbf =  rbf_svc.predict(gram_test_rbf)
        #print(predict_rbf)
        #Evaluate RBF
        #f1_rbf = f1_score(Y_test, predict_rbf, average='weighted')
        #print(f1_rbf)
        #Jaccard score is related with accuracy
        #jaccard_rbf = jaccard_score(Y_test, predict_rbf,average='weighted')
        #print(jaccard_rbf)
        #Accuracy RBF
        #certainty_rbf = metrics.accuracy_score(Y_test, predict_rbf)
        #print(certainty_rbf)
        #Sort Best RBF Sigmoidal results according to F1_Score
        #if len(Best_comb_svm)<50:
         #   Best_comb_svm[comb]=[f1_rbf,jaccard_rbf,certainty_rbf]
        #if len(Best_comb_svm)==50:
         #   sorted_comb_dict = sorted(Best_comb_svm.items(), key=operator.itemgetter(1))
          #  lowest_comb=sorted_comb_dict[49][0]
           # lowest_f1_value=Best_comb_svm[lowest_comb][0]
            #print (lowest_f1_value)
            #if f1_rbf>lowest_f1_value:
             #   del Best_comb_svm[lowest_comb]
              #  Best_comb_svm[comb]=[f1_rbf,jaccard_rbf,certainty_rbf]
        ##SVM WITH POLYNOMIAL (poly)
        #poly_svc = svm.SVC(kernel='poly')
        #print(poly_svc)
        # polynomial kernel computation
        #gram_train_poly = np.dot(X_train, X_train.T)
        #poly_svc.fit(gram_train_poly,Y_train)
        # predict on training examples
        #gram_test_poly = np.dot(X_test, X_train.T)
        #predict_poly =  poly_svc.predict(gram_test_poly)
        #print(predict_poly)
        #Evaluate Polynomial
        #f1_poly = f1_score(Y_test, predict_poly, average='weighted')
        #print(f1_poly)
        #Jaccard score is related with accuracy
        #jaccard_poly = jaccard_score(Y_test, predict_poly,average='weighted')
        #print(jaccard_poly)
        #Accuracy POLY
        #certainty_poly = metrics.accuracy_score(Y_test, predict_poly)
        #print(certainty_poly)
        #Sort Polynomial SVM Best Results according to F1_score
        #if len(Best_comb_svm)<210:
         #   Best_comb_svm[comb]=[f1_poly,jaccard_poly,certainty_poly]
        #if len(Best_comb_svm)==210:
         #   sorted_comb_dict = sorted(Best_comb_svm.items(), key=operator.itemgetter(1),reverse=False)
          #  lowest_comb=sorted_comb_dict[209][0]
           # lowest_f1_value=Best_comb_svm[lowest_comb][0]
            #print (lowest_f1_value)
            #if f1_poly>lowest_f1_value:
             #   del Best_comb_svm[lowest_comb]
              #  Best_comb_svm[comb]=[f1_poly,jaccard_poly,certainty_poly]
            #print (sorted_comb_dict[0:2])
        ##SVM WITH SIGMOID
        sigmoid_svc = svm.SVC(kernel='sigmoid')
        #print(sigmoid_svc)
        # sigmoidal kernel computation
        gram_train_sigmoid = np.dot(X_train, X_train.T)
        sigmoid_svc.fit(gram_train_sigmoid,Y_train)
        # predict on training examples
        gram_test_sigmoid = np.dot(X_test, X_train.T)
        predict_sigmoid =  sigmoid_svc.predict(gram_test_sigmoid)
        #print(predict_sigmoid)
        #Evaluate Sigmoid:
        f1_sigmoid = f1_score(Y_test, predict_sigmoid, average='weighted')
        #print(f1_sigmoid)
        #Jaccard score is related with accuracy
        jaccard_sigmoid = jaccard_score(Y_test, predict_sigmoid,average='weighted')
        #print(jaccard_sigmoid)
        #Accuracy Sigmoid
        certainty_sigmoidal = metrics.accuracy_score(Y_test, predict_sigmoid)
        #print(certainty_sigmoidal)
        #Write Results in File
        #file_output.write(str(comb)+","+str(f1_sigmoid)+","+str(jaccard_sigmoid)+","+str(certainty_sigmoidal)+"\n")
        Best_comb_svm[comb]=[f1_sigmoid,jaccard_sigmoid,certainty_sigmoidal]
        #Sort Sigmoidal SVM Results according to F1_score:
        if len(Best_comb_svm)<50:
           Best_comb_svm[comb]=[f1_sigmoid,jaccard_sigmoid,certainty_sigmoidal]
        if len(Best_comb_svm)==50:
            sorted_comb_dict = sorted(Best_comb_svm.items(), key=operator.itemgetter(1),reverse=False)
            lowest_comb=sorted_comb_dict[49][0]
            lowest_f1_value=Best_comb_svm[lowest_comb][0]
            #print (lowest_f1_value)
            if f1_sigmoid>lowest_f1_value:
                del Best_comb_svm[lowest_comb]
                Best_comb_svm[comb]=[f1_sigmoid,jaccard_sigmoid,certainty_sigmoidal]
            #print (sorted_comb_dict[0:2])

    #READ MORE ABOUT THIS 
    def best_features (self,input_file):
        #Select Best Columns - Attempt at using best_features model to select best columns
        df = pd.read_csv(input_file)
        df.head()
        X, y = load_digits(return_X_y=True)
        X.shape
        X_new = SelectKBest(chi2, k=20).fit_transform(X, y)
        print(X_new[0])
        print(len(X_new[0]))
        X_new.shape
        #print(X_new.shape)

    def select_best_columns (self,input_file):
        #Function to calculate all possible combinations from database
        #Select Best Columns
        #Select Best Columns - Attempt at using best_features model to select best columns
        df = pd.read_csv(input_file)
        #os.remove("RESULTS/RESULTS_KNN_25_75_4.csv")
        #file_output=open("RESULTS/RESULTS_KNN_25_75_4.csv",'a+')
        #file_output.write("Analysed Columns,Mean accuracy, k, Accuracy train, Accuracy test, Model\n")
        #os.remove("RESULTS/ListOfColumns.csv")
        #file_output=open("RESULTS/ListOfColumns.csv",'a+')
        #os.remove("RESULTS/Test10Features.pickle")
        a=list(df.columns.values)
        b=a[25:35]
        #Second Attempt at itertools
        c_cols = combinations(b,4)
        count_comb=0
        #Save in CSV code
        #for list_of_cols in c_cols:
            #count_comb+=1
            #file_output.write(str(count_comb)+","+str(list_of_cols)+"\n")
            #pickle.dump(str(count_comb)+","+str(list_of_cols)+"\n", open("RESULTS/List.pickle", 'a+'))
        pickle_output = open("RESULTS/Test10Features.pickle", "wb")
        pickle.dump(c_cols, pickle_output)
        pickle_output.close()
        pickle_infile = open("RESULTS/Test10Features.pickle",'rb')
        dict_model_comb = pickle.load(pickle_infile)
        #dict_model_comb_list=list(dict_model_comb)
        pickle_infile.close()
        print ("Pickle file with data created at: "+str(datetime.now()))
        #print ("The number of rows in the file is: "+len(list(c_cols)))
        #df.to_pickle(file_output)
        #object = pd.read_pickle("RESULTS/List.pickle")
        #for line in dict_model_comb:
         #   count_comb+=1
          #  #print (str(count_comb)+"-"+str(line))
        #print ("The number of combinations in the file is: "+str(count_comb)) 
        #RM_ESTAGIO_MACHINE_LEARNING.k_nearest_neighbors (input_file, list_of_cols,file_output)
        #df.head()
        #print(df.head())
        #for col in df:
           # print(col)
          #  print(len(col))

    #TEST SVM 10 Features
    def svm_best_features (self,input_file):
        input_file="RESULTS/Test10Features.pickle"
        file_output=open("RESULTS/RESULTS_SVM_Sigmoidal_50.csv",'w+')
        count_comb=0
        pickle_infile = open(input_file,'rb')
        dict_model_comb = pickle.load(pickle_infile)
        #dict_model_comb_list=list(dict_model_comb)
        pickle_infile.close()
        #os.remove("RESULTS/RESULTS_SVM.csv")
        #file_output=open("RESULTS/RESULTS_SVM.csv",'a+')
        for comb in dict_model_comb:
            count_comb+=1
            RM_ESTAGIO_MACHINE_LEARNING.svm("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/DB_BIOFILMS_PA.csv",comb)
            #print (str(count_comb)+"-"+str(line))
        print ("The number of combinations in the file is: "+str(count_comb)) 
        print (len(Best_comb_svm))
        for best_comb_model in Best_comb_svm:
            all_scores=Best_comb_svm[best_comb_model]
            file_output.write(str(best_comb_model)+","+str(all_scores)+"\n")

    #TEST NN 10 Features
    def nn_best_features (self,input_file):
        input_file="RESULTS/Test10Features.pickle"
        file_output=open("RESULTS/RESULTS_NN_50.csv",'w+')
        count_comb=0
        pickle_infile = open(input_file,'rb')
        dict_model_comb = pickle.load(pickle_infile)
        #dict_model_comb_list=list(dict_model_comb)
        pickle_infile.close()
        #os.remove("RESULTS/RESULTS_SVM.csv")
        #file_output=open("RESULTS/RESULTS_SVM.csv",'a+')
        for comb in dict_model_comb:
            count_comb+=1
            RM_ESTAGIO_MACHINE_LEARNING.neural_networks("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/DB_BIOFILMS_PA.csv",comb)
            #print (str(count_comb)+"-"+str(line))
        print ("The number of combinations in the file is: "+str(count_comb)) 
        print (len(Best_comb_svm))
        for best_comb_model in Best_comb_svm:
            all_scores=Best_comb_svm[best_comb_model]
            file_output.write(str(best_comb_model)+","+str(all_scores)+"\n")

    #TEST KNN 10 Features
    def knn_best_features (self,input_file):
        input_file="RESULTS/Test10Features.pickle"
        file_output=open("RESULTS/RESULTS_KNN_210.csv",'w+')
        count_comb=0
        pickle_infile = open(input_file,'rb')
        dict_model_comb = pickle.load(pickle_infile)
        #dict_model_comb_list=list(dict_model_comb)
        pickle_infile.close()
        #os.remove("RESULTS/RESULTS_SVM.csv")
        #file_output=open("RESULTS/RESULTS_SVM.csv",'a+')
        for comb in dict_model_comb:
            count_comb+=1
            RM_ESTAGIO_MACHINE_LEARNING.k_nearest_neighbors("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/DB_BIOFILMS_PA.csv",comb)
            #print (str(count_comb)+"-"+str(line))
        print ("The number of combinations in the file is: "+str(count_comb)) 
        print (len(Best_comb_svm))
        for best_comb_model in Best_comb_svm:
            all_scores=Best_comb_svm[best_comb_model]
            file_output.write(str(best_comb_model)+","+str(all_scores)+"\n")
            #file_output.write(str(mean_acc.argmax()+1)+","+str(best_comb_model)+","+str(all_scores)+"\n")
             
# Chamar a classe
RM_ESTAGIO_MACHINE_LEARNING=RM_ESTAGIO_MACHINE_LEARNING()
#RM_ESTAGIO_MACHINE_LEARNING.ler_base_dados_biofilm("DB_BIOFILMS_PA.csv")
#RM_ESTAGIO_MACHINE_LEARNING.linear_regression_biofilm("DB_BIOFILMS_PA.csv")
#RM_ESTAGIO_MACHINE_LEARNING.logistic_regression_biofilm("DB_BIOFILMS_PA.csv")
#RM_ESTAGIO_MACHINE_LEARNING.k_nearest_neighbors("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/DB_BIOFILMS_PA.csv")
#RM_ESTAGIO_MACHINE_LEARNING.neural_networks("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/DB_BIOFILMS_PA.csv")
#RM_ESTAGIO_MACHINE_LEARNING.best_features("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/DB_BIOFILMS_PA.csv")
#RM_ESTAGIO_MACHINE_LEARNING.select_best_columns("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/DB_BIOFILMS_PA.csv")
#RM_ESTAGIO_MACHINE_LEARNING.svm("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/INPUT_FILES/DB_BIOFILMS_PA.csv")
RM_ESTAGIO_MACHINE_LEARNING.svm_best_features("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/RESULTS/Test10Features.pickle")
#RM_ESTAGIO_MACHINE_LEARNING.nn_best_features("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/RESULTS/Test10Features.pickle")
#RM_ESTAGIO_MACHINE_LEARNING.knn_best_features("/Users/Rita Magalhaes/source/repos/Estágio_JC_1/PythonApplication2/RESULTS/Test10Features.pickle")


#PRINT FIRST ROW TO CHECK COLUMNS NAMES
#with open('Full_DW_withDescriptors_Full_Padel.csv', newline='') as f:
 # for row in f:
  #  print(row)
   # break
