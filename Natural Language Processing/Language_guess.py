#import packages.

import pandas as pd
import numpy as np
import nltk as lang
from sklearn.feature_extraction.text import CountVectorizer as tkzn
import seaborn as sns
import pickle #save trained classifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn import svm as svmpkg
from sklearn.model_selection import cross_val_score

text_data=pd.read_csv(r'C:\Users\Pabi\OneDrive\ML Projects\lang_data.csv')
#remove all observations with NaN values in the data set
text_data=text_data.dropna(axis=0)
n_obs,n_vars=text_data.shape
# we use 0.7 of the data for training
#we use 0.3 of the data for testing
train_data=text_data.iloc[:int(n_obs*0.7),:].copy(deep=True)
test_data=text_data.iloc[int(n_obs*0.7):n_obs,:].copy(deep=True)
n_train,p_train=train_data.shape
n_test,p_test=test_data.shape


afrdut_train=train_data[train_data['language']!='English']
n_afrdut=afrdut_train.shape[0]
eng_train=train_data[train_data['language']=='English']
n_eng=eng_train.shape[0]


#Notice quite a large class imbalance
#There are a total of 1442 observations with english label
#There are a total of 490 observations with Afri/Dutch Label
indx=list(range(n_eng))
np.random.shuffle(indx)
eng_samples={}
eng_samples['eng_set1']=eng_train.iloc[indx[:n_afrdut],:]
eng_samples['eng_set2']=eng_train.iloc[indx[n_afrdut:2*n_afrdut],:]
eng_samples['eng_set3']=eng_train.iloc[indx[2*n_afrdut:],:]
#in retropspect randomisation may not be nessacary as the observations ?are? independent.
#dont hurt to be safe though.
#convert everything to lower case
#change weird symbols
#Get a idea of the most common words used for english text.
#we will not remove stop words.

#Look next to add more features.
#create the 3 classifiers
n_train1=pd.concat([afrdut_train,eng_samples['eng_set1']])
n_train1['language'] = n_train1['language'].replace(to_replace='Afrikaans',value='Germanic')
n_train1['language'] = n_train1['language'].replace(to_replace='Nederlands',value='Germanic')
n_train2=pd.concat([afrdut_train,eng_samples['eng_set2']])
n_train2['language'] = n_train2['language'].replace(to_replace='Afrikaans',value='Germanic')
n_train2['language'] = n_train2['language'].replace(to_replace='Nederlands',value='Germanic')
n_train3=pd.concat([afrdut_train,eng_samples['eng_set3']])
n_train3['language'] = n_train3['language'].replace(to_replace='Afrikaans',value='Germanic')
n_train3['language'] = n_train3['language'].replace(to_replace='Nederlands',value='Germanic')

#Train different classifiers
splitter_1=tkzn(analyzer='word')
splitter_2=tkzn(analyzer='word')
splitter_3=tkzn(analyzer='word')
#fit transforms learns the vocab from raw text and constructs the corresponding bag of words matrix.
train_data_features1 = splitter_1.fit_transform(n_train1['text'])
train_data_features2 = splitter_2.fit_transform(n_train2['text'])
train_data_features3 = splitter_3.fit_transform(n_train3['text'])



num_clfs=int(n_eng/n_afrdut)+1
#Train the linear SVC
#Use cross validation to find the best c value
#range of 50 to 1000 is used.
c_range=[50,100,250,500,750,1000]
svm_CVerr1=[]
svm_CVerr2=[]
svm_CVerr3=[]
for penalty in c_range:
    svm_clf1=svmpkg.LinearSVC(C=penalty)
    scores1 = cross_val_score(svm_clf1,train_data_features1,n_train1['language'] ,cv=3)
    svm_CVerr1.append(np.mean(scores1))
    svm_clf2=svmpkg.LinearSVC(C=penalty)
    scores2 = cross_val_score(svm_clf2,train_data_features2,n_train2['language'] ,cv=3)
    svm_CVerr2.append(np.mean(scores2))
    svm_clf3=svmpkg.LinearSVC(C=penalty)
    scores3 = cross_val_score(svm_clf3,train_data_features3,n_train3['language'] ,cv=3)
    svm_CVerr3.append(np.mean(scores3))

best_c1=c_range[np.argmin(svm_CVerr1)]
finalsvm_clf1=svmpkg.LinearSVC(C=best_c1)
finalsvm_clf1.fit(train_data_features1,n_train1['language'])
best_c2=c_range[np.argmin(svm_CVerr2)]
finalsvm_clf2=svmpkg.LinearSVC(C=best_c2)
finalsvm_clf2.fit(train_data_features2,n_train2['language'])
best_c3=c_range[np.argmin(svm_CVerr3)]
finalsvm_clf3=svmpkg.LinearSVC(C=best_c3)
finalsvm_clf3.fit(train_data_features3,n_train3['language'])


#======================================================================================================================
#Over sample the minority class
#Make 9 smaller data with balanced labels in each data set.
dut=afrdut_train[afrdut_train['language']!='Afrikaans']
afri=afrdut_train[afrdut_train['language']!='Nederlands']
dut_n_obs=dut.shape[0]
afri_n_obs=afri.shape[0]

indx_2=list(range(afri_n_obs))
np.random.shuffle(indx_2)
afri_samples={}
max_classifiers=int(afri_n_obs/dut_n_obs)   #construct floor(afri_n_obs/dut_n_obs) classifiers
for k in range(max_classifiers):
    afri_samples['afri_set{}'.format(k)]=afri.iloc[indx_2[k*dut_n_obs:(k+1)*dut_n_obs]]
    afri_samples['train{}'.format(k)]=pd.concat([afri_samples['afri_set{}'.format(k)],dut])

#Train 9 SVMs
#train using word tolkenization.
afri_dut_clf_dict={}
for k in range(max_classifiers):
   afri_dut_clf_dict['afridut_crossval_err{}'.format(k)]=[]
   afri_dut_clf_dict['afridut_gramcrossval_err{}'.format(k)]=[]
   afri_dut_clf_dict['splitter_{}'.format(k)]=tkzn(analyzer='word')
   afri_dut_clf_dict['ngram_split_{}'.format(k)]=tkzn(analyzer='char_wb',ngram_range=(1,3))
   afri_dut_clf_dict['afridut_numeric{}'.format(k)]=afri_dut_clf_dict['splitter_{}'.format(k)].fit_transform(afri_samples['train{}'.format(k)]['text'])
   afri_dut_clf_dict['afridut_gramnumeric{}'.format(k)]=afri_dut_clf_dict['ngram_split_{}'.format(k)].fit_transform(afri_samples['train{}'.format(k)]['text'])

   for penalty in c_range:
       afridutsvm_clf=svmpkg.LinearSVC(C=penalty)
       scores = cross_val_score(afridutsvm_clf,
       afri_dut_clf_dict['afridut_numeric{}'.format(k)],
                          afri_samples['train{}'.format(k)]['language'] ,cv=2)
       afri_dut_clf_dict['afridut_crossval_err{}'.format(k)].append(np.mean(scores))
   best_c=c_range[np.argmax(afri_dut_clf_dict['afridut_crossval_err{}'.format(k)])]
   afri_dut_clf_dict['linsvmclf_{}'.format(k)]=svmpkg.LinearSVC(C=best_c)
   afri_dut_clf_dict['linsvmclf_{}'.format(k)].fit(afri_dut_clf_dict['afridut_numeric{}'.format(k)],
                                                afri_samples['train{}'.format(k)]['language'])


#======================================================================================================================

#Train 9 SVMs
#train using ngram tolkenization.
for k in range(max_classifiers):
    for penalty in c_range:
        afridutsvm_clf=svmpkg.LinearSVC(C=penalty)
        scores = cross_val_score(afridutsvm_clf,
                                 afri_dut_clf_dict['afridut_gramnumeric{}'.format(k)],
                                 afri_samples['train{}'.format(k)]['language'] ,cv=2)
        afri_dut_clf_dict['afridut_gramcrossval_err{}'.format(k)].append(np.mean(scores))
    best_c=c_range[np.argmax(afri_dut_clf_dict['afridut_gramcrossval_err{}'.format(k)])]
    afri_dut_clf_dict['lingramsvmclf_{}'.format(k)]=svmpkg.LinearSVC(C=best_c)
    afri_dut_clf_dict['lingramsvmclf_{}'.format(k)].fit(afri_dut_clf_dict['afridut_gramnumeric{}'.format(k)],
                                                        afri_samples['train{}'.format(k)]['language'])


#===============================================================================================================



#===============================================================================================================

# A function for deciding a winner between voting classfiers
#Takes in a matrix of votes
#Each row of the matrix corresponds votes from different classifiers
#Returns array of winner vote for each row.
def most_common(input_array):
    n_test_obs,n_votes=input_array.shape
    maj_vote=[]
    for k in range(n_test_obs):

        votes=input_array[k,:]
        winner=np.unique(votes,return_counts=True)[0][0]
        maj_vote.append(winner)
    return maj_vote

#Function for calcualting the accuary rate
#Takes in two arrays- the predicted labels and the actual labels

def calc_accuary(predict,actual):
    match=0
    n_obs=predict.shape[0]
    for k in range(n_obs):
        if predict[k]==actual[k]:
            match=match+1
    return match/n_obs


#function of implementing the model architecture
#Descripted in write up
#Returns a tuple consisting  of predicted values,actual values and accuray rate given test data
def DAGensemblesvm(test_data_2):
   #English vs Germainic Classification
    numeric_test_data1=splitter_1.transform(test_data_2['text'])
    numeric_test_data2=splitter_2.transform(test_data_2['text'])
    numeric_test_data3=splitter_3.transform(test_data_2['text'])
    pred1=finalsvm_clf1.predict(numeric_test_data1)
    pred2=finalsvm_clf2.predict(numeric_test_data2)
    pred3=finalsvm_clf3.predict(numeric_test_data3)
    engpred_matrix=pred1
    engpred_matrix=np.vstack((engpred_matrix,pred2))
    engpred_matrix=np.vstack((engpred_matrix,pred3))
    eng_maj_vote=most_common(np.transpose(engpred_matrix))
    eng_maj_vote_2=eng_maj_vote.copy()
    eng_maj_vote=np.asarray(eng_maj_vote,dtype=object)
    eng_maj_vote_2=np.asarray(eng_maj_vote_2)

    #English vs Germanic Classification complete
    #Next to take all instances of Germanic Classification and classify as Afrikaans or Dutch

    afri_dut_test_set=test_data_2.iloc[np.where(eng_maj_vote=='Germanic')]
    for k in range(max_classifiers):
        #I choose to use the ngram tolkenization instead of word as it leads to higher accuary
        afri_dut_clf_dict['ADgramnumeric_test_data{}'.format(k)]=afri_dut_clf_dict['ngram_split_{}'.format(k)].transform(afri_dut_test_set['text'])
        afri_dut_clf_dict['grampredicted{}'.format(k)]=afri_dut_clf_dict['lingramsvmclf_{}'.format(k)].predict(afri_dut_clf_dict['ADgramnumeric_test_data{}'.format(k)])
    grampred_matrix=np.transpose(afri_dut_clf_dict['grampredicted1'].copy())
    for k in range(1,max_classifiers):
        grampred_matrix=np.vstack((grampred_matrix,afri_dut_clf_dict['grampredicted{}'.format(k)]))
    grampred_matrix=np.transpose(grampred_matrix)
    grampredicted=most_common(grampred_matrix)
    grampredicted=np.asarray(grampredicted)
    afri_indx=np.asarray(np.where(eng_maj_vote_2=='Germanic'))

    for count in range(np.size(afri_indx)):
        pos=afri_indx[0,count]
        eng_maj_vote[pos]=grampredicted[count]
    predic_accuary=calc_accuary(eng_maj_vote,test_data_2['language'].as_matrix())

    return  eng_maj_vote,test_data_2['language'].as_matrix(),predic_accuary

votes1=DAGensemblesvm(test_data.copy(deep=True))
predict_languages=votes1[0]
actual_languages=votes1[1]
accuracy=votes1[2]

pickle_list=list([eng_samples,n_train1,n_train2,n_train3,splitter_1,splitter_2,splitter_3,
                  train_data_features1,train_data_features2,train_data_features3,
                  finalsvm_clf1,finalsvm_clf2,finalsvm_clf3,afri_samples,
                  afri_dut_clf_dict,predict_languages,actual_languages,accuracy])

outFile=open('pickle_trained_model.txt','wb')
pickle.dump(pickle_list,outFile)
