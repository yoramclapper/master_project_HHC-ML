import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor
from sklearn import neighbors
import xgboost as xgb
import sklearn.metrics as sklmetrics
import pickle
import scipy.stats as stats
#from sklearn.metrics import mean_absolute_percentage_error
from scipy.spatial import ConvexHull
import schedule
import gomea
import copy
from evaluate_routes import *
from akkerman_functions import *
from new_feature_functions import *
from point_pattern_features import *

#==============================================================================
# YC: load data set
#==============================================================================

rcount=0 # YC: no idea what this represents

# YC: sets 'ins' parameters to class attributes
class Inst:
    def __init__(self, ins):
        self.tasks = ins['tasks']
        self.shifts = ins['shifts']
        self.n = ins['n']
        self.d = ins['d']
        self.p = ins['p']
        self.tw = ins['tw']
        self.v = ins['v']
        self.ss = ins['ss']
        self.u = ins['u']
        self.Q = ins['Q']
        self.feasibleShiftsForClients = ins['feasibleShiftsForClients']
        
# YC: output measures of schedule
metrics = ['total_waiting_time', 'total_travel_time','total_overtime','total_idle_time']

#data_indices = [26,27,28]
# data_indices = [33,34] # only morning tasks, with half the nr of tasks
#data_indices = [35,36,37] # all customers have same tw length
data_indices = [0,1,2] # YC: data set indices of newly generated data (8-5-2023)
for data_indx in data_indices:
    #dataset_str = 'dataset_gom' + str(data_indx) + '.pickle'
    dataset_str = 'dataset_vrp' + str(data_indx) + '.pickle' # YC: data set name of newly generated data (8-5-2023)
    with open(dataset_str, 'rb') as handle:
        data = pickle.load(handle)
    # if dataset_str == 'dataset_gom{}.pickle'.format(data_indices[0]):
    #     dataset = data.copy()
    # else:
    #     dataset['instance'] += data['instance']
    #     for metric in metrics:
    #         dataset[metric] += data[metric]
    
    # YC: to load newly generated data within framework of test_env module 
    # it is probably necessary to concatenate data at each index (8-5-2023)
    dataset = data.copy() 
    
del data
instances = [Inst(dataset['instance'][i]['instance']) for i in range(len(dataset['instance']))]
dataset.pop('instance')
dataset.pop('info')

#==============================================================================
# YC: calculate features and add to data
#==============================================================================

# add Akkerman features
f_list = ["nr_cust", "area_encl_rec", "peri_encl_rec", "area_convx_hull", "peri_convx_hull", 
          "width_encl_rec", "height_encl_rec", "avg_dist", "avg_dist_depot-cust", 
          "dist_depot-rec_centr", "dist_depot-cust_centr", "avg_dist_cust-rec_centr", 
          "avg_dist_cust-cust_centr", "angle1", "angle2", "angle3", "geo_var1", "geo_var2", 
          "geo_var3", "geo_var4", "geo_var5", "geo_var6", "radius1", "radius2", "radius3", 
          "radius4", "radius5", "radius6", "rec_part1", "rec_part2", "rec_part3", 
          "rec_part4", "rec_part5", "rec_part6", "rec_part7", "rec_part8", 'total demand', 
          'avg demand', 'var demand', 'dem/veh cap', 'nr shifts', 'max demand/veh cap', 
          'nr high dem recs', 'avg dist in clusters', 'avg dist centroids', 
          'avg dist to clust center', 'avg dist clusters-depot', 'dist depot-furthest cust']

f = {i+1: f_list[i] for i in range(48)}
     
new_features = [
                'mean tw length', 'std tw length',
                'non_zero_in_D', 'non_zero_in_D2', 'sum_D', 'sum_D2', 'extra_dist_from_nn',
                'unclusterable', 'window spread',
                'space_time_dist_corr', 'tw midpoints std early', 'tw midpoints std mid', 'tw midpoints std late',
                'Local FCFS waiting', 'Local FCFS overtime', 'Local FCFS travel', #'FCFS total',
                'Global FCFS waiting', 'Global FCFS overtime', 'Global FCFS travel',# 'FCFS total2',
                'Load based FCFS wait', 'Load based FCFS overtime', 'Load based FCFS travel',# 'FCFS total3'
                'Period wise FCFS waiting', 'Period wise FCFS overtime', 'Period wise FCFS travel',# 'FCFS total3'
                'MMs wait early', 'MMs wait mid', 'MMs wait late',
                'GGs wait early', 'GGs wait mid', 'GGs wait late',
                'early load', 'mid load', 'late load',
                'service time std early', 'service time std mid', 'service time std late',
                'service time mean early', 'service time mean mid', 'service time mean late'
                ]

mei_features = ['ANNI','standard_distance','Pielou','asymp_pred','area','mean_centre_dist']
exclude_yoram_features = ['distance_avg', 'distance_std', 'nr_shifts','service_time_avg','service_time_std']

for feature in new_features:
    dataset[feature] = []
for feature in instances[0].tasks.task_features:
    if feature not in exclude_yoram_features:
        dataset[feature] = []
for feature in instances[0].shifts.shift_features:
    if feature not in exclude_yoram_features:
        dataset[feature] = []
for feature in f_list:
    dataset[feature] = []
for feature in mei_features:
    dataset[feature] = []

indx=0
tw_routes = []
shift_overtimes = []

# Extract features from instances and make dataframe
for ins in instances:    
    task_features = ins.tasks.task_features
    shift_features = ins.shifts.shift_features
    for task_key in task_features:
        if task_key not in exclude_yoram_features:
            dataset[task_key] += [task_features[task_key]]
    for shift_key in shift_features:
        if shift_key not in exclude_yoram_features:
            dataset[shift_key] += [shift_features[shift_key]]
    
    akk_feature, clusters = get_akkerman_features(ins, only_clust=False)
    # clusters = get_akkerman_features(ins, only_clust=True)
    for i in range(1,49):
        dataset[f[i]].append(akk_feature[i]) 
    
    tw_lengths = [tw[1]-tw[0] for tw in ins.tasks.time_window]
    dataset['mean tw length'].append( np.mean( tw_lengths ) )
    dataset['std tw length'].append(np.std(tw_lengths))
    tpm = trans_prob_matrix(ins)
    distances = np.array(ins.tasks.travel_matrix)[1:,1:]
    nn_matrix = near_neigh_matrix(distances)
    tpmmetrics = tpm_metrics(tpm,ins,nn_matrix)
    dataset['non_zero_in_D'].append( tpmmetrics[0] )
    dataset['non_zero_in_D2'].append( tpmmetrics[1] )
    dataset['sum_D'].append( tpmmetrics[2] )
    dataset['sum_D2'].append( tpmmetrics[3] )
    dataset['extra_dist_from_nn'].append( tpmmetrics[4] )
    dataset['unclusterable'].append( chain_metrics(clusters, distances, tpm, nn_matrix) )
    dataset['window spread'].append( time_window_spread(ins) )
    dataset['space_time_dist_corr'].append( tw_spatial_spread(ins) ) 
    
    routes2 = wt_route_split(ins)
    routes = FCFS_route(clusters,ins)
    routes3 = load_based_split(ins)
    routes4 = FCFS_by_period(ins)
    fcfs = eval_routes(routes, ins)
    fcfs2 = eval_routes(routes2, ins)
    fcfs3 = eval_routes2(routes3, ins)
    fcfs4 = eval_routes(routes4, ins)
    dataset['Local FCFS waiting'].append( fcfs['waiting'] )
    dataset['Local FCFS overtime'].append( fcfs['overtime'] )
    dataset['Local FCFS travel'].append( fcfs['travel'] )
    dataset['Global FCFS waiting'].append( fcfs2['waiting'] )
    dataset['Global FCFS overtime'].append( fcfs2['overtime'] )
    dataset['Global FCFS travel'].append( fcfs2['travel'] )
    dataset['Load based FCFS wait'].append( fcfs3['waiting'] )
    dataset['Load based FCFS overtime'].append( fcfs3['overtime'] )
    dataset['Load based FCFS travel'].append( fcfs3['travel'] )
    dataset['Period wise FCFS waiting'].append( fcfs4['waiting'] )
    dataset['Period wise FCFS overtime'].append( fcfs4['overtime'] )
    dataset['Period wise FCFS travel'].append( fcfs4['travel'] )
    
    period_indx = get_tw_in_period(ins)
    S = np.array(ins.tasks.service_time)
    for period in ['early', 'mid', 'late']:
        dataset['MMs wait '+period].append( MMs_wait(ins,period) )
        dataset['GGs wait '+period].append( GGs_wait(ins,period) )
        dataset['tw midpoints std ' + period].append( tw_midpoints_std(ins, period) )
        dataset['service time std '+ period].append( np.nan_to_num( np.std(S[period_indx[period]]) ) )
        dataset['service time mean '+ period].append( np.nan_to_num( np.mean(S[period_indx[period]]) ) )
        dataset[period+' load'].append( load_in_period(ins, period) )
    tw_routes.append( tw_of_routes(ins,routes) )
    shift_overtimes.append(fcfs['shift overtimes'])
    indx+=1
    
    # Point pattern features
    area = ins.tasks.region['width']*ins.tasks.region['height']
    dataset["area"] += [area]
    dataset["asymp_pred"] += [np.sqrt(area*ins.tasks.nr_tasks)]
    ANND_exp = 1/(2*np.sqrt(area*ins.tasks.nr_tasks))
    ANND_obs = np.mean(nearest_dist_list(ins.tasks.location))
    dataset["ANNI"] += [ANND_obs/ANND_exp]
    point_density = ins.tasks.nr_tasks/area
    dataset["Pielou"] += [point_density * math.pi * omega(ins)]
    centre = 0.5*ins.tasks.region["width"], 0.5*ins.tasks.region["height"]
    dataset["standard_distance"] += [standard_dist(ins.tasks.location)] 
    dataset["mean_centre_dist"] += [distance(centre, mean_center(ins.tasks.location))]
    
    if indx%100==0:
        print('Feature processing is at instance', indx)
    #print(indx)


#==============================================================================
# YC: setup dataframe from dataset for analysis
#==============================================================================

indx_akkerman_zero = [9,10,11,14,19,23,24,29,30,33,34,37,38,39,40,42,43,47,48]
columns_to_be_deleted = [f[i] for i in indx_akkerman_zero] 
df = pd.DataFrame(dataset) 
df = df.drop(columns_to_be_deleted, axis=1)


#label data
features_metrics = list(df)
features = []

for attribute in features_metrics:
    if attribute not in metrics:
        features += [attribute]
df = df[features+metrics]

df.insert(0, 'ins index', [i for i in range(len(instances))])
# dff = df.copy()
# keep_features = features
# features_copy = features.copy()
# for feat in features_copy:
#     if feat in f_list+features_tw_delete:
#         features.remove(feat)
# features = keep_features
# df = dff[['ins index']+features+[metric]]
    
# features_tw_delete = ['mean tw length', 'std tw length','GGs wait early', 'GGs wait mid', 'GGs wait late',
#                 'early load', 'mid load', 'late load',
#                 'service time std early', 'service time std mid', 'service time std late',
#                 'service time mean early', 'service time mean mid', 'service time mean late']

#
# imp = ['std tw length',
#  'extra_dist_from_nn',
#  'Local FCFS travel',
#  'Global FCFS waiting',
#  'Global FCFS travel',
#  'Load based FCFS wait',
#  'Load based FCFS travel',
#  'MMs wait early',
#  'GGs wait early',
#  'late load',
#  'asymp_pred']

# features_delete = ['Local FCFS travel', 'Load based FCFS travel', 'Global FCFS travel', 'extra_dist_from_nn']#, 'asymp_pred', 'Global FCFS waiting', 'Load based FCFS wait']
# for feat in features:
#     if feat not in imp:
#         features.remove(feat)


# environment to train and test ML model on generated data
metric = metrics[2] # [wait, trav, over, idle]

            
#==============================================================================
# YC: define ML methods and prepare data
#==============================================================================

def linear_regression(X_train,y_train,X_test):
    model = LinearRegression()
    reg = model.fit(X_train, y_train)
    return reg.predict(X_test), model.coef_, model

def neural_net(X_train,y_train,X_test, hidden_layers, activation='identity'):
    regr = MLPRegressor(max_iter = 1000, hidden_layer_sizes=hidden_layers).fit(X_train, y_train)
    return regr.predict(X_test)

def random_forest(X_train,y_train,X_test):
    regr = RandomForestRegressor()
    regr.fit(X_train, y_train)
    return regr.predict(X_test), regr.feature_importances_, regr

def xgboost(X_train,y_train,X_test):
    xgb_model = xgb.XGBRegressor(objective="reg:linear")
    xgb_model.fit(X_train, y_train)
    return xgb_model.predict(X_test), None, None
            

#create input and target data            
X = df[['ins index']+features].values
Y = df[metric].values

#create train and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 100)


# YC: first classify if waiting time is zero or positive 
# followed by prediciting positives if I recall correctly
TwoStageML = False
if TwoStageML:
    # First stage ML 
    X_positive = X[np.where(Y>0)]
    Y_positive = Y[np.where(Y>0)]
    y_test_binary = y_test.copy()
    y_train_binary = y_train.copy()
    y_test_binary[np.nonzero(y_test_binary)] = 1 
    y_train_binary[np.nonzero(y_train_binary)] = 1 
    
    classifier = neighbors.KNeighborsClassifier().fit(X_train[:,1:], y_train_binary)
    y_pred_binary = classifier.predict(X_test[:,1:])
    print(sklmetrics.classification_report(y_test_binary,y_pred_binary))
    
    # Extract training instances with positive waiting time
    
    X_train_positive = X_train[np.where(y_train>0)]
    y_train_positive = y_train[np.where(y_train>0)]
    X_test_positive  = X_test[np.where(y_pred_binary==1)]
    y_pred = y_pred_binary.copy() # no need to copy



#==============================================================================
# YC: results and displays
#==============================================================================

#benchmark
print("")
print("benchmark of predicting {} with mean".format(metric))
y_pred_benchmark = [np.mean(y_train) for i in range(len(y_test))]
print('MAE:',sklmetrics.mean_absolute_error(y_test, y_pred_benchmark))
print('WAPE:', np.sum(abs(y_test-y_pred_benchmark))/np.sum(y_test))
print("")


    
if False:
    predictor = neural_net
    hidden_layer_list = [(100,), (50,50), (34,33,33), (25,25,25,25), (20,20,20,20,20)]
    for hidden_layers in hidden_layer_list:
        #y_pred, importance, model = predictor(X_train, y_train, X_test)
        y_pred = predictor(X_train[:,1:], y_train, X_test[:,1:], hidden_layers)
        #results = permutation_importance(model, X_train, y_train, scoring='neg_mean_squared_error')
        #importance = results.importances_mean
        print("Model performance")
        print('MAE:',sklmetrics.mean_absolute_error(y_test, y_pred))
        print('WAPE:', np.sum(abs(y_test-y_pred))/np.sum(y_test))
        print('hidden layers:', hidden_layers, "\n")
    
for predictor in [linear_regression, xgboost, random_forest]:
    # y_pred[np.where(y_pred_binary==1)], importance, model = predictor(X_train_positive[:,1:], y_train_positive, X_test_positive[:,1:])
    y_pred, importance, model = predictor(X_train[:,1:], y_train, X_test[:,1:])
    #y_pred,_,_ = predictor(X_train, y_train, X_test)
    # results = permutation_importance(model, X_train[:,1:], y_train, scoring='neg_mean_squared_error')
    #importance = results
    
    print("Model performance:")
    print('MAE:',sklmetrics.mean_absolute_error(y_test, y_pred))
    print('WAPE:', np.sum(abs(y_test-y_pred))/np.sum(y_test))
    print('MAPE:', sklmetrics.mean_absolute_percentage_error(y_test,y_pred))
    print(predictor, "\n")

    #######  linear regression analysis ####################
    if predictor == linear_regression:
        coef = model.coef_
        coef_dict = {features[i]: coef[i] for i in range(len(features))}
        coef_dict['intercept'] = model.intercept_
        # for feature in coef_dict:
        #     print(feature, coef_dict[feature])
        print("")
        print("R2 score:", sklmetrics.r2_score(y_test, y_pred))
    #######################################################

################ Plots #######################################
res = pd.DataFrame({'ins index': X_test[:,0], 'Real Values':y_test, 'Predicted Values':y_pred, 'abs error': abs(y_pred-y_test)})
res = res.sort_values(by='Real Values')    
res.insert(0, 'plot index', [i for i in range(len(y_pred))])
res2 = res.sort_values(by='abs error',ascending=False)
plt.plot(res['Real Values'].values, 'bo', markersize=2, label='Actual')
plt.plot(res['Predicted Values'].values, 'ro', markersize=2, label='Predicted')
plt.xlabel('Instances sorted by real waiting time')
plt.ylabel('waiting')
plt.legend()
plt.tight_layout()
plt.savefig(metric[6:]+' vrptw all feat except periodwise.png', dpi=300)
plt.show()


important_features = []
importance_of_imp_feat = []
show_feature_importance = True
if show_feature_importance:
    for i in range(len(importance)):
        if importance[i]>0.015:
          important_features.append(features[i])
          importance_of_imp_feat.append(importance[i])
    # plot feature importance
    plt.bar(important_features, importance_of_imp_feat)
    plt.xticks(
        rotation=45, 
        horizontalalignment='right',
        fontweight='light',
        fontsize=8 
    )
    plt.ylabel("Relative feature importance")
    plt.tight_layout()
    # plt.savefig(metric[6:]+" feature importance vrptw all feat except periodwise.png", dpi=300)
    plt.show()
    print('len imp', len(important_features))
    # plt.hist(TSP_constant)
    # print("\n mean TSP constant: ", np.mean(TSP_constant))
    # print("actual constant (Figliozzi): ", 0.765)

# Correlation matrix
# imp_df = {'feature': important_features, 'score':importance_of_imp_feat}
# imp_df = pd.DataFrame(imp_df)
# sorted_imp_df = imp_df.sort_values(by='score',ascending=False)
# top_10 = sorted_imp_df['feature'].values.tolist()[:10]
# df2 = df[top_10]
# corr_matrix = df2.corr()
# plt.figure(figsize=(10,6))
# sns.heatmap(corr_matrix,annot=True)
# plt.tight_layout()
# plt.savefig("corr matrix.png", dpi=300)
#  plot features
# inss = [0,20,50,799]
# colors = ['r','g','c','y']
# track_ins = np.array(df_big_waiting['ins index'])[inss].tolist()
# for feature in ['Period wise FCFS waiting']:#features[:3]:
#     feature_values = np.array(df[feature])
#     metric_values = np.array(df[metric])
#     plt.plot(df[feature], df[metric], 'o', markersize=2)
#     for i in range(4):
#         plt.plot(feature_values[track_ins[i]], metric_values[track_ins[i]], colors[i]+'o', markersize=5)
#     plt.xlabel(feature)
#     plt.ylabel('waiting time')
#     plt.show()

# for i in range(len(instances)):
#     if feature_values[i]>600 and metric_values[i]<50:
#         print(i)

# plt.plot(df['Local FCFS travel'], df['extra_dist_from_nn'], 'bo')

