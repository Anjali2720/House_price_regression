import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.base import BaseEstimator , TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb


train = pd.read_csv('train.csv')
test_fin = pd.read_csv('test.csv')

'''
# Basic introduction to the dataset
for i in train.columns:
 null_v = train[i].isnull().sum()
 if null_v >0:
  print (i,' has null=', null_v)

for i in train.columns:
  if train[i].dtype == 'object':
   print (i,'is an object with unique values','\n',train[i].value_counts())
      
'''   
xtr = pd.concat([train,test_fin])

#Transforming target variable so that ~ Normally distributed 
xtr['SalePrice'] = np.log(xtr['SalePrice'])

#tr,te = train_test_split(train,test_size = 0.2, random_state = 3)


trs = xtr.copy() 
trs2 = xtr.copy()

'''
#checking correlations
corr_mat = trs.corr()
sorter = corr_mat['SalePrice'].sort_values(ascending =False)
	   
#Plotting highest corr variable 
vars = ('OverallQual','GrLivArea','TotalBsmtSF','1stFlrSF','GarageArea','2ndFlrSF')

for i in vars:
 plt.scatter(x = trs[i] , y = trs['SalePrice'], alpha = 0.3)
 plt.xlabel(i)
 plt.ylabel('sale price')
 #plt.show()
'''
# Deleting Outliers

#print(trs[trs['GrLivArea']<4000])
#print (trs[trs['GarageArea']>1200])
'''
trs.drop([523], inplace = True)
trs.drop([1061],inplace = True)
trs.drop([1190],inplace = True)
'''

# Identify highly null values  

for i in trs.columns:
 nul = trs[i].isnull().sum()
 if nul>0:
  prop_null= (i,float(nul)/float(len(trs)))
  #print(prop_null)
highly_null = ['Alley','PoolQC','Fence','MiscFeature','FireplaceQu']

class DataWrangle(BaseEstimator,TransformerMixin):
 def __init__(self,drop_too = True):
  self.drop_too= drop_too
  
 def fit(self,trs):
  return self
 def transform(self,trs): 
 #dropping highly null values
  
  trs.drop(['Id','Alley','PoolQC','Fence','MiscFeature','FireplaceQu'],axis =1,inplace =True)

  #identify and change dtype of those num which are cat var but have int values
  cat_col =[]
  num_col=[]
  for i in trs.columns:
   if trs[i].dtype == 'object':
    cat_col.append(i) 
   else:
    num_col.append(i)
    num_cat = [] 
  for i in num_col:
   uni = trs[i].unique()  
   if len(uni)<20:
    #print(i,len(uni))
    num_cat.append(i)

  yr_col = ['YearBuilt','YearRemodAdd','GarageYrBlt','MoSold','YrSold']
  for i in yr_col:
   num_cat.append(i)
 
  trs[num_cat] = trs[num_cat].astype('object') 

  #remove useless cat var :useless means which don't offer diversity
  cat_col = cat_col+num_cat
  freq_list = []
  for i in cat_col:
   if trs[i].isnull().sum()>0:
    trs[i].fillna('None')
   freq = trs[i].value_counts().max()
   freq = float(freq)/float(len(trs))
   freq_list.append([i,freq])                #freq_list contains proportional count of unique items
 
  freq_d = pd.DataFrame(np.array(freq_list),columns = ['name','freq'])
  freq_d.set_index(freq_d['name'],inplace = True)
  freq_d = freq_d['freq'].sort_values(ascending = False)
  col = freq_d.head(20).index                              
  #for i in col:
   #df = pd.DataFrame(trs[i].value_counts())
   #sns.barplot(x = df.index,y= df[i])
   #plt.xlabel(i)
   #plt.show()
   #print(trs[i].value_counts())
  trs.drop(['Utilities', 'PoolArea', 'Street', 'Condition2', 
  'LowQualFinSF', '3SsnPorch', 'RoofMatl', 'Heating', 'MiscVal', 
  'KitchenAbvGr', 'BsmtHalfBath', 'LandSlope', 'CentralAir', 
  'Functional', 'PavedDrive', 'GarageCond', 'Electrical', 
  'GarageQual', 'LandContour', 'BsmtCond'],axis =1,inplace = True)   #these variables have very high proportion of repeated values for observations
  
  #FEATURE ENGINEERING (kept only useful features)
  trs['totalsf'] = trs['1stFlrSF']+trs['2ndFlrSF']+trs['TotalBsmtSF']
  #trs['room/sf'] = trs['TotRmsAbvGrd'].astype('float')/trs['flrsf'].astype('float')
  #trs['totalbath'] = trs['HalfBath']+trs['BsmtFullBath']+trs['FullBath']+trs['BsmtHalfBath']
  #trs['Gr+o'] = trs['GrLivArea']+trs['totalsf']
  trs['GarageArea/car'] = trs['GarageArea'].astype('float')/trs['GarageCars'].astype('float')
  trs['GarageArea/car'].fillna(0,inplace =True)
  trs['youth'] = trs['YrSold']-trs['YearRemodAdd']

  trs.drop('1stFlrSF',axis =1,inplace= True)
  trs.drop('ScreenPorch',axis=1,inplace=True)#after permutation importance
  return trs
 
dw =DataWrangle()
trs =dw.transform(trs)

salepr = trs['SalePrice']  
trs.drop('SalePrice',axis =1,inplace = True)
#print(trs.columns)

def transforming_var(trs):
 #Applying LabelEncoder
 str_cat= []
 num_cate = []
 for i in trs.columns:
  if trs[i].dtype =='object':
   unique_ivar = (trs[i].unique()).tolist()
   if type(unique_ivar[1])==str:
    trs[i].fillna('3',inplace  = True)
    str_cat.append(i)
   else:
    trs[i].fillna(0,inplace = True)
    num_cate.append(i)
    #print(i)
 cat_v = str_cat+num_cate
 for i in str_cat:
  le = LabelEncoder()
  df = trs[i]
  trs[i] = le.fit_transform(df)
 #Applying Standardisation
 num_var = []
 for i in trs.columns:
  if i not in cat_v:
   num_var.append(i)
   trs[i].fillna(0,inplace =True)
   sc = StandardScaler()
   trs[num_var]=sc.fit_transform(trs[num_var])
 
 return trs,cat_v


trs,cat_v=transforming_var(trs)


salepr.fillna(0,inplace=True)
trs.drop('SalePrice',axis=1,inplace=True)

#TRAINING

trax,valx,tray,valy = train_test_split(trs,salepr,test_size = 0.4,random_state=9)



param = {
'application':'regression',
'boosting' :'gbdt',
#'num_iterations':3500,
'learning_rate' : 0.01,#0.005 #0.01
#'max_depth': 1000,
'verbose':1
}

d_tr = lgb.Dataset(trax,tray)
d_val = lgb.Dataset(valx,valy,reference = d_tr)

eval_result = {}

gbm = lgb.train(param,d_tr,num_boost_round = 1000,
valid_sets = [d_tr,d_val],
evals_result = eval_result,
verbose_eval = 10,early_stopping_rounds=50)

ax = lgb.plot_metric(eval_result,metric='l2')
#plt.show()

def score(X,y):
 y_pred = gbm.predict(X)
 return np.sqrt(mean_squared_error(y,y_pred))

import eli5
from eli5.permutation_importance import get_score_importances
base_score,score_decreases = get_score_importances(score,trs.to_numpy(),salepr.to_numpy())
feature_importances = np.mean(score_decreases,axis =0)
fe_dic = {}

for i,fea_n in enumerate(trs.columns):
 fe_dic[fea_n]=feature_importances[i]

print(sorted(fe_dic.items(),key=lambda x:x[1])) 

'''
#Feature importance with GBM
m = gbm.feature_name()
n =gbm.feature_importance()
a= zip(n,m)
print(m,sorted(a,reverse = True ))

lgb.plot_importance(gbm)
plt.show()
'''

#All the included features are important except  ('ScreenPorch', 0.0) removing it (inside DataWrangle)


test_d = dw.transform(test_fin)
test_d,_ = transforming_var(test_d)
print('test columns',test_d.columns)

print(test_d.columns==trax.columns)

pred = gbm.predict(test_d)
pred = np.exp(pred)
