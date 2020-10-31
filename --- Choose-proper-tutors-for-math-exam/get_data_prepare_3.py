# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 17:21:23 2020

@author: rost_
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
#import matplotlib.pyplot as plt
#import seaborn as sns
#import matplotlib

#from catboost import CatBoostClassifier
#from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


#from scipy.spatial.distance import euclidean
#from tqdm.notebook import tqdm, trange
#from sklearn.model_selection import KFold


global X_train, cat_features


class BetaEncoder:
    '''BetaEncoder
    BetaEncoder is used to encode categorical features with a beta-binomial conjugate pair
    model (i.e. a beta posterior predictive distribution for binary target). 
    For each categorical feature, this object stores a beta (y==0) and alpha (y==1)
    column with a row for each existing level of the categorical feature.  
    The input to fit() should be an array-like of 1,0 for y
    and array-like of strings for X.
    The output of transform() will be <column>__[M] where [M] is a particular moment 
    of the beta distribution [‘mvsk’], m and v are default.
    By default, a prior of alpha=.5, beta=.5 (uninformative) is used.
    Note: transform() takes the optional argument `training` (bool) for which
    y must be supplied. This results in total_count being decremented and 
    alpha or beta being decremented depending on y value.
    Parameters
    ----------
    beta_prior (float): prior for beta. default = .5
    alpha_prior (float):  prior for alpha. default = .5
    random_state (integer): random state for bootstrap samples. default = 1
    n_samples (integer): number of bootstrap samples. default = 100
    Attributes
    ----------
    _beta_prior (float) - prior for beta. default = .5
    _alpha_prior (float) - prior for alpha. default = .5
    _random_state (integer): random state for bootstrap samples. default = 1
    _n_samples (integer): number of bootstrap samples. default = 100
    _beta_distributions (dict) - houses the categorical beta distributions
        in pandas dataframes with cols `alpha` and `beta`
    Methods
    ----------
    fit()
    transform()
    Examples
    --------
    >>>import pandas as pd
    >>>from sklearn.datasets import load_boston
    >>>from sklearn.model_selection import train_test_split
    >>>from beta_encoder import BetaEncoder
    >>>bunch = load_boston()
    >>>y = bunch.target
    >>>X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>>categorical_cols=['CHAS', 'RAD']
    >>>X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)
    >>>encoder = BetaEncoder()
    >>>encoder.fit(X_train, y_train, columns=categorical_cols)
    >>>#print out the beta parameters for each level
    >>>encoder._beta_distributions
    >>>#transform the training dataset (leave one out)
    >>>encoder.transform(X_train, y=y_train, training=True, columns=categorical_cols)
    >>>#transform the test columns (just a pure link and fill na with prior)
    >>>encoder.transform(X_test, columns=categorical_cols)
    '''
    def __init__(self, alpha=0.5, beta=0.5, n_samples=10, sample_size=.75, random_state=1):
        '''init for BetaEncoder
        Args:
            alpha - prior for number of successes
            beta - prior for number of failures
        '''
        # Validate Types
        if type(alpha) != float:
            raise AttributeError("Argument 'alpha' must be of type float")
        if type(beta) is not float:
            raise AttributeError("Argument 'beta' must be of type float")
        if type(sample_size) is not float:
            raise AttributeError("Argument 'sample_size' must be of type float")
        if type(n_samples) is not int:
            raise AttributeError("Argument 'n_samples' must be of type int")
        if type(random_state) is not int:
            raise AttributeError("Argument 'random_state' must be of type int")

        #Assign
        self._alpha_prior = alpha
        self._beta_prior = beta
        self._beta_distributions = dict()
        self._random_state = random_state
        self._n_samples = n_samples
        self._sample_size = sample_size
        np.random.seed(random_state)

        
        
    def fit(self, X, y, columns=None):
        '''fit
        Method to fit self.beta_distributions
        from X and y
        Args:
            X (array-like) - categorical columns
            y (array-like) - target column (1,0)
            columns (list of str) - list of column names to fit
                otherwise, attempt to fit just string columns
        Returns:
            beta_distributions (dict) - a dict of pandas DataFrame for each
                categorical column with beta and alpha for each level
        '''
        if len(X) != len(y):
            print("received: ",len(X), len(y))
            raise AssertionError("Length of X and y must be equal.")

        X_temp = X.copy(deep=True)
        categorical_cols = columns
        if not categorical_cols:
            categorical_cols = self.get_string_cols(X_temp)

        #add target
        target_col = '_target'
        X_temp[target_col] = y       
        self.target = y.copy()
        

        for categorical_col in categorical_cols:

            ALL_LEVELS = X_temp[[categorical_col, target_col]].groupby(categorical_col).count().reset_index()
            
                
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
            for train_idx, valid_idx in skf.split(X_temp, y):
                X_sample = X_temp[[categorical_col, target_col]]
                X_sample = X_sample.iloc[train_idx, :]

                
                #full count (alpha + beta)
                full_count = X_sample[[categorical_col, target_col]].groupby(categorical_col).count().reset_index()
                full_count = full_count.rename(index=str, columns={target_col: categorical_col+"_full_count"})

                #alpha
                positive_count = X_sample[[categorical_col, target_col]].groupby(categorical_col).sum().reset_index()
                positive_count = positive_count.rename(index=str, columns={target_col: categorical_col+"_positive_count"})
                
                #merge them
                temp = pd.merge(full_count, positive_count, on=[categorical_col])
                temp['_alpha'] = self._alpha_prior + temp[categorical_col+"_positive_count"]
                temp['_beta'] = self._beta_prior + temp[categorical_col+"_full_count"] - temp[categorical_col+"_positive_count"]

                #fill NAs with prior
                temp = pd.merge(ALL_LEVELS, temp, on=categorical_col, how='left')
                temp['_alpha'] = temp['_alpha'].fillna(self._alpha_prior)
                temp['_beta'] = temp['_beta'].fillna(self._beta_prior)

                if categorical_col not in self._beta_distributions.keys():
                    self._beta_distributions[categorical_col] = temp[[categorical_col,'_alpha','_beta']]
                else:
                    self._beta_distributions[categorical_col][['_alpha','_beta']] += temp[['_alpha','_beta']]

            # report mean alpha and beta:
            self._beta_distributions[categorical_col]['_alpha'] = self._beta_distributions[categorical_col]['_alpha']/self._n_samples
            self._beta_distributions[categorical_col]['_beta'] = self._beta_distributions[categorical_col]['_beta']/self._n_samples
        return

    
    
    def transform(self, X, moments='v', columns=None):
        '''transform
        Args:
            X (array-like) - categorical columns matching
                the columns in beta_distributions
            columns (list of str) - list of column names to transform
                otherwise, attempt to transform just string columns
            moments (str) - composed of letters [‘mvsk’] 
                specifying which moments to compute where ‘m’ = mean, 
                ‘v’ = variance, ‘s’ = (Fisher’s) skew and ‘k’ = (Fisher’s) 
                kurtosis. (default=’m’)
        '''
        X_temp = X.copy(deep=True)
        categorical_cols = columns
        if not categorical_cols:
            categorical_cols = self.get_string_cols(X_temp)


        for categorical_col in categorical_cols:
            if categorical_col not in self._beta_distributions.keys():
                raise AssertionError("Column "+categorical_col+" not fit by BetaEncoder")

            #add `_alpha` and `_beta` columns vi lookups, impute with prior
            X_temp = X_temp.merge(self._beta_distributions[categorical_col], on=[categorical_col], how='left')

            X_temp['_alpha'] = X_temp['_alpha'].fillna(self._alpha_prior)
            X_temp['_beta'] = X_temp['_beta'].fillna(self._beta_prior)         
            
            #   encode with moments
            if 'm' in moments:
                X_temp[categorical_col+'__M'] = X_temp["_alpha"]/(X_temp["_alpha"]+X_temp["_beta"])
            if 'v' in moments:
                X_temp[categorical_col+'__V'] = (X_temp["_alpha"]*X_temp["_beta"]) / \
                   (((X_temp["_alpha"]+X_temp["_beta"])**2)*(X_temp["_alpha"]+X_temp["_beta"]+1))
            if 's' in moments:                
                alpha = X_temp["_alpha"]
                beta  = X_temp["_beta"]
                num = 2*(beta-alpha)*np.sqrt(alpha+beta+1)
                dem = (alpha+beta+2)*np.sqrt(alpha*beta)                
                X_temp[categorical_col+'__S'] = num/dem
                
            if 'k' in moments:               
                alpha = X_temp["_alpha"]
                beta  = X_temp["_beta"]
                num = 6*(alpha-beta)**2*(alpha+beta+1) - alpha*beta*(alpha+beta+2)
                dem = alpha*beta*(alpha+beta+2)*(alpha+beta+3)                                        
                X_temp[categorical_col+'__K'] = num/dem                               
               
            if 'w' in moments:                                
#                glob_mean = self.target.mean()
                glob_bad  = self.target.sum()
                glob_good = self.target.count()-self.target.sum()               
                bad_DB  = X_temp["_alpha"]/glob_bad
                good_DB = X_temp["_beta"]/glob_good                
                
                tempo = (good_DB/bad_DB)
                WOE = np.log(tempo)
                iv = (good_DB-bad_DB)*WOE
                
                gg={}    
                for i in range(len(iv)):
                    gg[iv.index[i]] = iv.iloc[i]                                    
                X_temp[categorical_col+'__WOE'] = X_temp[categorical_col].map(gg)    
                X_temp[categorical_col+'__WOE'] = iv.values
                
            #and drop columns
            X_temp = X_temp.drop([categorical_col], axis=1)
            X_temp = X_temp.drop(["_alpha"], axis=1)
            X_temp = X_temp.drop(["_beta"], axis=1)

        return X_temp

    def get_string_cols(self, df):
        idx = (df.applymap(type) == str).all(0)
        return df.columns[idx]




class get_mod_data():
    
    def __init__(self, for_whom='log_regression'):
        
        self.for_whom         = for_whom              
        self.lesson_price_ratio         = None      
        self.age_ratio                  = None         
        self.mean_exam_points_ratio     = None
        self.years_of_experience_ratio  = None
        

    

    def data_prepare(self, df_train, df_test):
        
       
        df_trn = df_train.copy()
        df_tst = df_test.copy()   
                
        
        
        df_tst.loc[(df_tst['mean_exam_points'] == 32) , 'mean_exam_points'] = 33  
           
        df_tst.loc[(df_tst['lesson_price'] == 3800) , 'lesson_price'] = 3750      
        df_tst.loc[(df_tst['lesson_price'] == 3850) , 'lesson_price'] = 3750        
        df_tst.loc[(df_tst['lesson_price'] == 300) , 'lesson_price'] = 350      
       
    
        df_tst['mean_exam_points'] = np.log(df_tst['mean_exam_points'])  
        df_trn['mean_exam_points'] = np.log(df_trn['mean_exam_points'])      
        df_tst['lesson_price'] = np.log(df_tst['lesson_price'])  
        df_trn['lesson_price'] = np.log(df_trn['lesson_price'])      
        df_tst['age'] = np.log(df_tst['age'])  
        df_trn['age'] = np.log(df_trn['age'])      
        df_tst['years_of_experience'] = np.log(df_tst['years_of_experience']+0.000001)  
        df_trn['years_of_experience'] = np.log(df_trn['years_of_experience']+0.000001)  
    


        
        def get_param(feature):
            
            bad_list=[]
            total_list=[]
                    
            a=df_trn.groupby(by=df_trn[feature])['choose'].agg(["count","sum"])
            a['total_local'] = 1
            a['bad_local'] = 0.000000001
                    
            skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)        
            for train_idx, valid_idx in skf.split(df_trn, df_trn.choose):
    
                df_trn_cv = df_trn.iloc[train_idx, :]
                b=df_trn_cv.groupby(by=df_trn_cv[feature])['choose'].agg(["count","sum"])            
                
                a['total_local'] = b['count']
                a['bad_local'] = b['sum']
                
                a['bad_local'] = a['bad_local'].fillna(0.)
                a['total_local'] =a['total_local'].fillna(0.)
                         
                bad_list.append(a['bad_local'].copy())
                total_list.append(a['total_local'].copy())
            
            
            
            a['bad_local'] = bad_list[0] 
            a['total_local'] = total_list[0]  
            for i in range(1, len(bad_list)):
                a['bad_local'] = a['bad_local']+bad_list[0]
                a['total_local'] = a['total_local']+total_list[0]   
            a['bad_local'] = a['bad_local']/len(bad_list)
            a['total_local'] = a['total_local']/len(bad_list)     
            
            
                        
            a.loc[(a['total_local'] == 0) , 'total_local'] = 1             
            a.loc[(a['bad_local'] == 0) , 'bad_local'] = 0.00001          
       
            a['bad_local_smooth']=a['bad_local'].rolling(window=7, center=True).mean()
            a['total_local_smooth']=a['total_local'].rolling(window=7, center=True).mean()     
            
            a['bad_local_smooth'].iloc[0] = a['bad_local_smooth'].iloc[3]
            a['bad_local_smooth'].iloc[1] = a['bad_local_smooth'].iloc[3] 
            a['bad_local_smooth'].iloc[2] = a['bad_local_smooth'].iloc[3]         
            a['bad_local_smooth'].iloc[-1] = a['bad_local_smooth'].iloc[-4]
            a['bad_local_smooth'].iloc[-2] = a['bad_local_smooth'].iloc[-4]
            a['bad_local_smooth'].iloc[-3] = a['bad_local_smooth'].iloc[-4]         
                
            a['total_local_smooth'].iloc[0] = a['total_local_smooth'].iloc[3]
            a['total_local_smooth'].iloc[1] = a['total_local_smooth'].iloc[3]  
            a['total_local_smooth'].iloc[2] = a['total_local_smooth'].iloc[3]          
            a['total_local_smooth'].iloc[-1] = a['total_local_smooth'].iloc[-4]       
            a['total_local_smooth'].iloc[-2] = a['total_local_smooth'].iloc[-4]               
            a['total_local_smooth'].iloc[-3] = a['total_local_smooth'].iloc[-4]                       
      
            a.loc[(a['total_local'] < 25) , 'bad_local'] = a['bad_local_smooth'] 
            a.loc[(a['total_local'] < 25) , 'total_local'] = a['total_local_smooth']         
            a.loc[(a['total_local'] < 25) , 'prop'] = np.log((a['bad_local_smooth']/(a['total_local_smooth']))+0.000001)        
            
            a['good_local'] = a['total_local'] - a['bad_local']  
            a.loc[(a['good_local'] == 0) , 'good_local'] = 0.00003          
            a['prop'] = np.log(a['bad_local']/(a['total_local'])+0.000001)                          
    
    
            return a['prop']
     


        # Обработаем числоыые признаки

#==============================================================================    
        
        
        ratio = get_param('lesson_price')  
        if self.lesson_price_ratio is None:
            self.lesson_price_ratio = ratio
        else:
            ratio = (self.lesson_price_ratio)


        
        df_trn['lesson_price_r'] = df_trn['lesson_price'].map(ratio)
        df_tst['lesson_price_r'] = df_tst['lesson_price'].map(ratio)   
        df_tst['lesson_price_r'] = df_tst['lesson_price_r'].fillna(0.000001)
                   
#==============================================================================    
        
        ratio = get_param('age')
        if self.age_ratio is None:
            self.age_ratio = ratio
        else:
            ratio = (self.age_ratio)
            
        
        df_trn['age_r'] = df_trn['age'].map(ratio)
        df_tst['age_r'] = df_tst['age'].map(ratio)    
        df_tst['age_r'] = df_tst['age_r'].fillna(0.000001)
          
#==============================================================================    
            
        ratio = get_param('mean_exam_points')
        if self.mean_exam_points_ratio is None:
            self.mean_exam_points_ratio = ratio
        else:
            ratio = (self.mean_exam_points_ratio)
        
        
        df_trn['mean_exam_points_r'] = df_trn['mean_exam_points'].map(ratio)
        df_tst['mean_exam_points_r'] = df_tst['mean_exam_points'].map(ratio)
        df_tst['mean_exam_points_r'] = df_tst['mean_exam_points_r'].fillna(0.000001)
             
#==============================================================================    
        
        ratio = get_param('years_of_experience')

        if self.years_of_experience_ratio is None:
            self.years_of_experience_ratio = ratio
        else:
            ratio = (self.years_of_experience_ratio)

        
        df_trn['years_of_experience_r'] = df_trn['years_of_experience'].map(ratio)
        df_tst['years_of_experience_r'] = df_tst['years_of_experience'].map(ratio)       
        df_tst['years_of_experience_r'] = df_tst['years_of_experience_r'].fillna(0.000001)
        
          
#==============================================================================    
        
        train_features = [f for f in df_trn if f not in ['choose', 'Id']]
        train_features_dig = [f for f in df_trn if f in ['age', 'lesson_price', 'mean_exam_points',
                                                        'years_of_experience',
                                                        'mean_exam_points_r', 'age_r', 'years_of_experience_r','lesson_price_r'
                                                        ]]    
      
        cat_features = [f for f in df_trn if f not in ['choose', 'Id','age','lesson_price', 'mean_exam_points',
                                                        'years_of_experience',
                                                        'mean_exam_points_r', 'age_r', 'years_of_experience_r','lesson_price_r']]    
     



        df_trn[cat_features] = df_trn[cat_features].astype('str')
        df_tst[cat_features] = df_tst[cat_features].astype('str')
        
        
        df_trn[cat_features] = df_trn[cat_features].astype('object')    
        df_trn[train_features_dig] = df_trn[train_features_dig].astype('float32')   
        df_trn[cat_features] = df_trn[cat_features].astype('object')
           
        
        df_tst[cat_features] = df_tst[cat_features].astype('object')    
        df_tst[train_features_dig] = df_tst[train_features_dig].astype('float32')   
        df_tst[cat_features] = df_tst[cat_features].astype('object')    
        
     
        
        y_train=df_trn['choose'].copy()
        X_train = df_trn.copy()
        X_test  = df_tst.copy()


        # Обработаем категориальные признаки
    
#==============================================================================    
#==============================================================================    
    
    
           
        X_train_M = X_train.copy()
        X_test_M  = X_test.copy()
        X_train_W = X_train.copy()    
        X_test_W  = X_test.copy()     
        
        cat_features = [f for f in X_train if f not in ['choose', 'Id','age','lesson_price', 'mean_exam_points',
                                                        'years_of_experience',
                                                        'mean_exam_points_r', 'age_r', 'years_of_experience_r','lesson_price_r']]    
            
        encoder = BetaEncoder()
        encoder.fit(X_train, y_train.copy(), columns=cat_features)
        encoder._beta_distributions
        #transform the training dataset 
        X_train=encoder.transform(X_train, columns=cat_features,  moments='v')
        #transform the test columns 
        X_test=encoder.transform(X_test, columns=cat_features,  moments='v')    
        
        
        encoder = BetaEncoder()
        encoder.fit(X_train_M, y_train.copy(), columns=cat_features)
        encoder._beta_distributions
        #transform the training dataset 
        X_train_M=encoder.transform(X_train_M, columns=cat_features,  moments='m')
        #transform the test columns 
        X_test_M=encoder.transform(X_test_M, columns=cat_features,  moments='m')        
        
        
        encoder = BetaEncoder()
        encoder.fit(X_train_W, y_train.copy(), columns=cat_features)
        encoder._beta_distributions
        #transform the training dataset 
        X_train_W=encoder.transform(X_train_W, columns=cat_features,  moments='w')
        #transform the test columns 
        X_test_W=encoder.transform(X_test_W, columns=cat_features,  moments='w')         
    
        
        
        cat_features = [f for f in X_train_M if f not in ['choose', 'Id','age','lesson_price', 'mean_exam_points',
                                                        'years_of_experience',
                                                        'mean_exam_points_r', 'age_r', 'years_of_experience_r','lesson_price_r']]    
        X_train[cat_features] = X_train_M[cat_features]
        X_test[cat_features]  = X_test_M[cat_features]  
        
    
           
    
        cat_features = [f for f in X_train_W if f not in ['choose', 'Id','age','lesson_price', 'mean_exam_points',
                                                        'years_of_experience',
                                                        'mean_exam_points_r', 'age_r', 'years_of_experience_r','lesson_price_r']]    




        X_train[cat_features] = X_train_W[cat_features]    
        X_test[cat_features]  = X_test_W[cat_features]        
        cat_features = [f for f in X_train if f not in ['choose', 'Id','age','lesson_price', 'mean_exam_points',
                                                        'years_of_experience',
                                                        'mean_exam_points_r', 'age_r', 'years_of_experience_r','lesson_price_r']]    
    
        X_train[cat_features]=np.log(X_train[cat_features]+0.0000001)
        X_test[cat_features] =np.log(X_test[cat_features]+0.0000001)       
        

#        if self.for_whom == 'booster':        
#        
##            X_train_exp = X_train.copy()
#            X_train['y'] = y_train.values  
#            pos_features=X_train[(X_train['y']>0)]   
#            print(pos_features.shape, X_train.shape)
#            for i in range(8):
#                X_train = pd.concat([X_train, pos_features], ignore_index=True, sort=False)
#                #print(i, 'step nan in X_train',X_train.isnull().sum(), X_train.shape)        
#            X_train.iloc[np.random.permutation(len(X_train))]
#            y_train = X_train['y']
#            X_train = X_train.drop(['y'], axis=1)
    
    
 
        #Выкинем лишнее
        
        X_train = X_train.drop(["age"], axis=1)
        X_test  = X_test.drop(["age"], axis=1)
        X_train = X_train.drop(["chemistry__V"], axis=1)
        X_test  = X_test.drop(["chemistry__V"], axis=1)
        X_train = X_train.drop(["chemistry__WOE"], axis=1)
        X_test  = X_test.drop(["chemistry__WOE"], axis=1)
        X_train = X_train.drop(["geography__WOE"], axis=1)
        X_test  = X_test.drop(["geography__WOE"], axis=1)
        X_train = X_train.drop(["geography__V"], axis=1)
        X_test  = X_test.drop(["geography__V"], axis=1)
        
       
        
#        print(X_train.columns)
        train_features = [f for f in X_train if f not in ['choose', 'Id']]       
        scaler = StandardScaler()   
    
        X_train[train_features] = scaler.fit_transform(X_train[train_features])
        X_test[train_features]  = scaler.transform(X_test[train_features])
    
        
        
#        print('Data shapes: ',X_train.shape, X_test.shape, len(train_features))
        
        return X_train, X_test, y_train, train_features    


