# -*- coding: utf-8 -*-
'''
Created on 2017��1��24��

@author: ZQZ
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from xgboost import plot_tree
from xgboost import plot_importance
from xgboost import XGBRegressor
from xgboost import XGBModel
from xgboost import XGBClassifier
from sklearn.cross_validation import train_test_split
from statsmodels.tools import eval_measures
from sklearn import cross_validation, metrics # metrics contains roc_curve and auc   #Additional scklearn functions
from collections import OrderedDict
from operator import itemgetter

# 0 - functions
def get_numpy_data(data,output):
    # prepend variable 'constant' to the features list
    new_col = [col for col in data.columns if output not in col]
    features_matrix = data[new_col].as_matrix()
    output_sarray = data[output]
    output_array = output_sarray.as_matrix()
    return(features_matrix, output_array)

def xgb_model_fit(dt_feature_list,feature_names,mAlg, trainDT,trainOutputDT,testDT,testOutputDT,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = mAlg.get_xgb_params()
        xgtrain = xgb.DMatrix(trainDT, label=trainOutputDT)
        #xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=mAlg.get_params()['n_estimators'], nfold=cv_folds, metrics='auc', early_stopping_rounds=early_stopping_rounds)
        mAlg.set_params(n_estimators=cvresult.shape[0])


    #Fit the algorithm on the data
    mAlg.fit(trainDT, trainOutputDT,eval_metric='auc')
    
    #Predict training set:
    train_prediction = mAlg.predict(trainDT)
    train_predprob = mAlg.predict_proba(trainDT)[:,1]
    
    #Predict test set:
    test_prediction = mAlg.predict(testDT)
    test_predprob = mAlg.predict_proba(testDT)[:,1]
    
        
    #Print model report:
    print( "\nModel Report : \n")
    print( "Accuracy (Train) : %.4g" % metrics.accuracy_score(trainOutputDT, train_prediction))
    print( "AUC Score (Train): %f" % metrics.roc_auc_score(trainOutputDT, train_predprob))
    print( "Confusion Matrix (Train) :", metrics.confusion_matrix(trainOutputDT, train_prediction))
    #print "AUC Score (Train): %f" % metrics.confusion_matrix(trainOutputDT, train_predprob)
    print( "True Negative : %i"% metrics.confusion_matrix(trainOutputDT, train_prediction)[0][0])
    print( "True Positive : %i"% metrics.confusion_matrix(trainOutputDT, train_prediction)[1][1])
    print( "False Negative : %i"% metrics.confusion_matrix(trainOutputDT, train_prediction)[1][0])
    print( "False Positive : %i"% metrics.confusion_matrix(trainOutputDT, train_prediction)[0][1])
    print( "Recall : %f"% (metrics.confusion_matrix(trainOutputDT, train_prediction)[1][1] / (metrics.confusion_matrix(trainOutputDT, train_prediction)[1][1] + metrics.confusion_matrix(trainOutputDT, train_prediction)[1][0])) )
    print( "Precision : %f"% (metrics.confusion_matrix(trainOutputDT, train_prediction)[1][1] / (metrics.confusion_matrix(trainOutputDT, train_prediction)[1][1] + metrics.confusion_matrix(trainOutputDT, train_prediction)[0][1])) )
    
    
    print('\n')
    print( "Accuracy(Test) : %.4g" % metrics.accuracy_score(testOutputDT, test_prediction))
    print( "AUC Score (Test): %f" % metrics.roc_auc_score(testOutputDT, test_predprob))
    print( "Confusion Matrix (Test) : " ,  metrics.confusion_matrix(testOutputDT, test_prediction))
    
    #print "AUC Score (Test): %f" % metrics.confusion_matrix(testOutputDT, test_predprob)
    print( "True Negative : %i"% metrics.confusion_matrix(testOutputDT, test_prediction)[0][0])
    print( "True Positive : %i"% metrics.confusion_matrix(testOutputDT, test_prediction)[1][1])
    print( "False Negative : %i"% metrics.confusion_matrix(testOutputDT, test_prediction)[1][0])
    print( "False Positive : %i"% metrics.confusion_matrix(testOutputDT, test_prediction)[0][1])
    print( "Recall : %f"% (metrics.confusion_matrix(testOutputDT, test_prediction)[1][1] / (metrics.confusion_matrix(testOutputDT, test_prediction)[1][1] + metrics.confusion_matrix(testOutputDT, test_prediction)[1][0])) )
    print( "Precision : %f"% (metrics.confusion_matrix(testOutputDT, test_prediction)[1][1] / (metrics.confusion_matrix(testOutputDT, test_prediction)[1][1] + metrics.confusion_matrix(testOutputDT, test_prediction)[0][1])) )
    
    #Plot Training ROC
    print('\n')
    ''' plot ROC_Training'''
    fpr_training, tpr_training, _ = metrics.roc_curve(trainOutputDT, train_predprob)
    roc_auc_training = metrics.auc(fpr_training, tpr_training)
    plt.figure()
    lw = 2
    plt.plot(fpr_training, tpr_training, color='darkorange',
         lw=lw, label='ROC curve (AUC/Area = %0.2f)' % roc_auc_training)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC_Traing')
    plt.legend(loc="lower right")
    plt.show()
    
     #Plot Testing ROC
    print('\n')
    ''' plot ROC_Testing'''
    fpr_testing, tpr_testing, _ = metrics.roc_curve(testOutputDT, test_predprob)
    roc_auc_testing = metrics.auc(fpr_testing, tpr_testing)
    plt.figure()
    lw = 2
    plt.plot(fpr_testing, tpr_testing, color='darkorange',
         lw=lw, label='ROC curve (AUC/Area = %0.2f)' % roc_auc_testing)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC_Testing')
    plt.legend(loc="lower right")
    plt.show()
    
    
    
    print('\n')
    ''' plot feature importance'''
    #for i in mAlg.feature_importances_ :
    #    print(i)          
    #print(mAlg.feature_importances_)
    
    ft_impor = {}
    ''' 
    for k,v in zip(dt_feature_list,mAlg.feature_importances_):
        ft_impor[k]=str('{0:.16f}'.format(v))
        
    ftdt = pd.DataFrame([ft_impor])
    ftdt.to_csv('ftimport_tissue_T_N.csv')
    '''
    ft_impor = OrderedDict(sorted(ft_impor.items(), key=itemgetter(1), reverse = True))
    print(list(ft_impor.items())[:5])
    print(list(ft_impor.keys())[:5])
    #ftdt = pd.DataFrame(ft_impor.items()[:15])
    #ftdt.to_csv('ftimport_tissue_T_N.csv')
    #=IF(COUNTIF(A:A,"A")>0,1,0)   
    
    ''''''
    feat_imp = pd.Series(mAlg._Booster.get_fscore())#.reindex(feature_names)#.sort_values(ascending=False)#.reindex(feature_names)
    
    feat_imp = feat_imp.to_frame(name='score')
    feature_new_names = [feature_names[int(str(x)[1:])] for x in list(feat_imp.index)]
    feat_imp['Proteins'] = np.array(feature_new_names)
    feat_imp = feat_imp.set_index('Proteins').sort_values(by=['score'], ascending=False)
    #print(feat_imp)
    feat_imp[:30].plot(kind='barh', title='Feature Importances', figsize=(10, 8))
    plt.ylabel('Feature Importance Score')
    plt.show()
    
    



# 1- modelling    
all_data = pd.read_csv('PPPA_without_missing_value_FOR_XGB.csv')#PPPA_without_missing_value_FOR_XGB PPPA_FOR_XGB
all_data = all_data.ix[:,2:]
all_data = all_data[(all_data['Tissue'] =='T') | (all_data['Tissue'] =='N')]
all_data['Tissue'] = all_data['Tissue'].apply(lambda x : 0 if x =='N' else 1)
all_data['Tissue'] = all_data['Tissue'].astype(int)

dt_columns_list = all_data.columns.tolist()
dt_feature_list = dt_columns_list[:dt_columns_list.index('CPP_ID')]
new_dt_feature_list = dt_feature_list + ['Tissue']
all_data = all_data[new_dt_feature_list]

print(all_data[all_data['Tissue'] == 0].shape)
print(all_data[all_data['Tissue'] == 1].shape)

feature_names = [str(x)[5:11] for x in list(all_data.columns)[:-1]]

all_data_X, all_data_y = get_numpy_data(all_data,'Tissue')

X_train, X_test, y_train, y_test = train_test_split(all_data_X,all_data_y,test_size =0.2, random_state=42)

xgb_t3 = XGBClassifier(learning_rate =0.01,
                       n_estimators=200,#200#100
                       max_depth=5,#5 #4
                       min_child_weight=10,
                       gamma=0,
                       reg_alpha=0.005, #0.001,
                       subsample=0.9,#0.7,
                       colsample_bytree=0.6,#0.55,
                       objective= 'binary:logistic',
                       nthread=4,
                       scale_pos_weight=1,#35.56, 1  
                       seed=27)


xgb_model_fit(dt_feature_list,feature_names, xgb_t3, X_train, y_train, X_test, y_test)
