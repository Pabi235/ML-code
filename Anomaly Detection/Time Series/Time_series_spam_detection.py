import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn import model_selection as mod_select
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE
import sys
import csv

#Import data.

#spam_data=pd.read_csv(r'...\spam_detection_data.csv',index_col='five_min_intervals',parse_dates=True)
spam_data=pd.read_csv(r'...\spam_detection_data.csv')
spam_data_2=pd.read_csv(r'...\spam_detection_data.csv')


spam_data_3=spam_data.copy(deep=True)

#Normalize the counts column.
# [x-mean()]/std dev
normalized_counts=(spam_data_3.counts-spam_data_3.counts.mean())/spam_data_3.counts.std()
normalized_counts = normalized_counts.rename(columns={'counts': 'normalized_counts'})
spam_data_normalized = pd.concat([spam_data_3, normalized_counts], axis=1)
spam_data_normalized.columns=['counts','five_min_intervals','normalized_counts']
spam_data_normalized['five_min_intervals']=pd.to_datetime(spam_data_normalized['five_min_intervals'])


spam_oct=spam_data_normalized.copy(deep=True)[54156:60784]
spam_aug=spam_data_normalized.copy(deep=True)[41997:47550]
spam_oct_2=spam_oct.copy(deep=True)
spam_data_3['five_min_intervals']=pd.to_datetime(spam_data_3['five_min_intervals'])

#Visulize data for different months.

sns.tsplot(data=spam_oct_2['counts'])
sns.tsplot(data=spam_oct['normalized_counts'])
sns.tsplot(data=spam_aug['normalized_counts'])

# some helper variables
n_rows,n_cols=spam_data.shape


#Want to create a data frame that spams over 3 hours of the data set.
#60/3=12 five minute inyterval obersvations per row
#New data will be of dimension [start_time,end_time,12_observations] X floor(N/12) oobservations
n_tuples=np.int(np.floor(n_rows/12))

three_hour_window=pd.DataFrame(index=range(0,n_tuples),columns=['start_time','end_time','t_1','t_2','t_3','t_4','t_5','t_6',
                                        't_7','t_8','t_9','t_10','t_11','t_12'])
three_hour_window['start_time']=pd.to_datetime(three_hour_window['start_time'])
three_hour_window['end_time']=pd.to_datetime(three_hour_window['end_time'])

date_start=0
date_end=11
for k in range(0,(n_tuples)):
    three_hour_window.loc[k,['start_time','end_time']]=spam_data_normalized.loc[[date_start,date_end],'five_min_intervals'].values

    three_hour_window.loc[
        k, ['t_1', 't_2', 't_3', 't_4', 't_5', 't_6', 't_7', 't_8', 't_9', 't_10', 't_11', 't_12']] = spam_data_normalized.loc[(k * 12):(k * 12 +11 ), 'counts'].values

    date_start, date_end = date_start + 12, date_end + 12

three_hour_window['start_hour']=pd.DatetimeIndex(three_hour_window['start_time']).hour
three_hour_window['end_hour']=pd.DatetimeIndex(three_hour_window['end_time']).hour


#perform the same with normalized counts

three_hour_window_normalized=pd.DataFrame(index=range(0,n_tuples),columns=['start_time','end_time','t_1','t_2','t_3','t_4','t_5','t_6',
                                        't_7','t_8','t_9','t_10','t_11','t_12'])
three_hour_window_normalized['start_time']=pd.to_datetime(three_hour_window_normalized['start_time'])
three_hour_window_normalized['end_time']=pd.to_datetime(three_hour_window_normalized['end_time'])

date_start=0
date_end=11
for k in range(0,(n_tuples)):
    three_hour_window_normalized.loc[k,['start_time','end_time']]=spam_data_normalized.loc[[date_start,date_end],'five_min_intervals'].values
    three_hour_window_normalized.loc[
        k, ['t_1', 't_2', 't_3', 't_4', 't_5', 't_6', 't_7', 't_8', 't_9', 't_10', 't_11', 't_12']] = spam_data_normalized.loc[k:(k+11), 'normalized_counts'].values
    date_start, date_end = date_start + 12, date_end + 12


three_hour_window_normalized['start_hour']=pd.DatetimeIndex(three_hour_window_normalized['start_time']).hour
three_hour_window_normalized['end_hour']=pd.DatetimeIndex(three_hour_window_normalized['end_time']).hour



#train /test split

n_rows_series,ncols_series=three_hour_window_normalized.shape

n_train=np.int(np.floor(n_rows_series*0.75))


three_hour_window.to_csv(r'...\three_hour_data.csv')


three_hour_window_copy=three_hour_window.copy(deep=True)
three_hour_window_normalized_copy=three_hour_window_normalized.copy(deep=True)



# Setting up Normal and Outlier data
spam_data=pd.read_csv(r'...\X_spam_detection.csv')


spam_outliers=spam_data.copy(deep=True)[spam_data.Sum_Outlier==1]
normal_data=spam_data.copy(deep=True)[spam_data.Sum_Outlier!=1]

normal_data_relevant_data=normal_data[['t_1', 't_2', 't_3', 't_4', 't_5', 't_6',
       't_7', 't_8', 't_9', 't_10', 't_11', 't_12', 'start_hour', 'end_hour',
       'Sum_over_interval', 'Max_in_interval', 'Num_large_values']]


outlier_data=spam_outliers[['t_1', 't_2', 't_3', 't_4', 't_5', 't_6',
       't_7', 't_8', 't_9', 't_10', 't_11', 't_12', 'start_hour', 'end_hour',
       'Sum_over_interval', 'Max_in_interval', 'Num_large_values']]

normal_train,normal_test=mod_select.train_test_split(normal_data_relevant_data,test_size=0.2, random_state=42)




#Visualizing the data

#Visualize the data using TSNE


tsne_data=normal_data.copy(deep=True)[['t_1', 't_2', 't_3', 't_4', 't_5', 't_6',
       't_7', 't_8', 't_9', 't_10', 't_11']]

tsne_data_withoutliers=spam_data.copy(deep=True)[['t_1', 't_2', 't_3', 't_4', 't_5', 't_6',
       't_7', 't_8', 't_9', 't_10', 't_11']]

tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=500)
tsne_results = tsne.fit_transform(tsne_data)
tsne_withoutliers=tsne.fit_transform(tsne_data_withoutliers)

tsne_x=tsne_results[:,0]
tsne_y=tsne_results[:,1]

tsne_withoutliers_X=tsne_withoutliers[:,0]
tsne_withoutliers_Y=tsne_withoutliers[:,1]




plt.scatter(tsne_x,tsne_y)
plt.scatter(tsne_withoutliers_X,tsne_withoutliers_Y,c=spam_data.Sum_Outlier)



# Training a one-class SVM

clf = svm.OneClassSVM(nu=0.1, kernel="sigmoid", gamma=0.015)
clf.fit(normal_train)
y_pred_train = clf.predict(normal_train)
y_pred_test = clf.predict(normal_test)

tsne_train = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=500)
tsne_fit_train = tsne_train.fit_transform(normal_train)
tsnetrain_X=tsne_fit_train[:,0]
tsnetrain_Y=tsne_fit_train[:,1]
plt.scatter(tsnetrain_X,tsnetrain_Y,c=y_pred_train)

tsne_test = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=500)
tsne_fit_test = tsne_test.fit_transform(normal_test)
tsnetest_X=tsne_fit_test[:,0]
tsnetest_Y=tsne_fit_test[:,1]
plt.scatter(tsnetest_X,tsnetest_Y,c=y_pred_test)


y_pred_test = clf.predict(normal_test)
y_pred_outliers = clf.predict(outlier_data)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_train=y_pred_train.size
n_error_outliers = y_pred_outliers[y_pred_outliers == -1].size
n_outlier=y_pred_outliers.size
n_test=y_pred_test.size


#fit classifier on entire dataset

clf_validation_data=spam_data.copy(deep=True)[['t_1', 't_2', 't_3', 't_4', 't_5', 't_6',
       't_7', 't_8', 't_9', 't_10', 't_11', 't_12', 'start_hour', 'end_hour',
       'Sum_over_interval', 'Max_in_interval', 'Num_large_values']]

tsne_test = TSNE(n_components=2, verbose=1, perplexity=60, n_iter=750,learning_rate=750)
tsne_fit_valid = tsne_test.fit_transform(clf_validation_data)
tsnevalid_X=tsne_fit_valid[:,0]
tsnevalid_Y=tsne_fit_valid[:,1]


clf_validation_labels=spam_data.copy(deep=True)['Sum_Outlier']
clf = svm.OneClassSVM(nu=0.15, kernel="sigmoid", gamma=0.00005)
clf_rfb=svm.OneClassSVM(nu=0.1,kernel='rbf',gamma=0.000000275)
clf.fit(normal_train)
clf_rfb.fit(normal_train)
clf_validation_pred_labels=clf.predict(clf_validation_data)
clf_rbf_validation_pred_labels=clf_rfb.predict(clf_validation_data)

plt.scatter(tsnevalid_X,tsnevalid_Y,c=clf_validation_labels)
plt.scatter(tsnevalid_X,tsnevalid_Y,c=clf_validation_pred_labels)
plt.scatter(tsnevalid_X,tsnevalid_Y,c=clf_rbf_validation_pred_labels)



spam_data['spam_predcition']=clf_rbf_validation_pred_labels



n_error_outliers/outlier_behave_data.shape[0]

# Training a one-class SVM for normalized data


clf_norm = svm.OneClassSVM(nu=0.17, kernel="rbf", gamma=0.05565646)
clf_norm.fit(norm_behave_data_normalized_train)
y_norm_pred_train = clf_norm.predict(norm_behave_data_normalized_train)
y_norm_pred_test = clf_norm.predict(norm_behave_data_normalized_test)
y_norm_pred_outliers = clf_norm.predict(outlier_behave_data_normalized)
n_norm_error_train = y_norm_pred_train[y_norm_pred_train == -1].size
n_norm_error_test = y_norm_pred_test[y_norm_pred_test == -1].size
n_norm_error_outliers = y_norm_pred_outliers[y_norm_pred_outliers == -1].size

n_norm_error_outliers/outlier_behave_data_normalized.shape[0]

norm_behave_data_train.to_csv(r'...\norm_behave_data_train.csv')
norm_behave_data_test
three_hour_window.to_csv(r'...\three_hour_data.csv')
norm_behave_data_test.to_csv(r'...\norm_behave_data_test.csv')
outlier_behave_data.to_csv(r'...\outlier_behave_data.csv')
#
# ts_plot_1=sns.tsplot(data=spam_small,time='five_min_intervals',value='counts')
# ts_plot=pd.Series(spam_small['counts'], index=pd.DatetimeIndex(spam_small['five_min_intervals']))
#
# timeseries_spamsmall=pd.Series(spam_small['counts'], index=pd.to_datetime(spam_small['five_min_intervals']))
# sns.tsplot(data=spam_small['counts'])
# timeseries_spam=pd.Series(spam_data['counts'], index=pd.to_datetime(spam_data['five_min_intervals']))
# sns.tsplot(data=spam_data['counts'])

spam_data.to_csv(r'...\X_spam_data_with_predictions.csv')
