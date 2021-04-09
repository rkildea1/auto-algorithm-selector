import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import sklearn
from sklearn import datasets,metrics,tree
from sklearn.model_selection import train_test_split
from sklearn import model_selection as cv # replaced the above line with this
from sklearn.svm import SVC

#import the file as a pandas df
url = "https://raw.githubusercontent.com/rkildea1/auto-algorithm-selector/main/student-mat.csv" ##if you change the set, you will need to update any of the column names declared elsewhere in the file


stalcDF = pd.read_csv(url,index_col=0)


#first thing to do is pair up the data headers with the data in the first row
myds = stalcDF.iloc[0].reset_index().apply(tuple, axis=1)  #write array col head & row1 to a dict_items
colHeader_row1_list = [] 
for key, value in myds:
    list_item = (key,value)
    colHeader_row1_list.append(list_item)

categorical_cols = [] #this will be the list of categorical attributes
numeric_cols = []     #this will be the list of numeric attributes

for i, x in colHeader_row1_list:
#     print((type(x)),i,x) # just checking what kind of format the data is in (e.g., int or numpy.int64)
    if type(x) != str: 
        numeric_cols.append(i)
    else:
        categorical_cols.append(i)
        
print("There are:", (len(categorical_cols)), "text columns")              
print("There are:", (len(numeric_cols)), "numeric columns")
                          
#convert text in the columns with text data to numeric form via label encoding..
for i in categorical_cols:
#     print(i)
    stalcDF[i] = stalcDF[i].astype("category")
    stalcDF[(i+"_CAT")]  = stalcDF[i].cat.codes #duplicate each column encoded, and add "_CAT" after it
#len(stalcDF.columns) 
stalcDF_N= stalcDF.copy()
stalcDF_N['Avg_Grade'] = stalcDF_N[['G1', 'G2','G3']].mean(axis=1) 
stalcDF_N['Avg_Alc_Cnsmptn'] = stalcDF_N[['Dalc', 'Walc']].mean(axis=1) 


                                # Create the Target variables (names and numeric version)
#create "dataset.target_names"
stalcDF_target_names = str(list(set(stalcDF_N["sex"]))) #<class 'str'>                               
#create "dataset.target"
stalcDF_target = stalcDF_N["sex_CAT"].values # <class 'numpy.ndarray'>



                              #Features i.e., the dataset class label values as an array

#I want this to only have numeric data so create a numeric-only dataset. I can do this by dropping all columns that 
#match the column name of my "categorical_cols" variable i created earlier


#drop highly correlated columns 
stalcDF_N.drop(['G1', 'G2','G3'], axis=1, inplace=True) #drop the 3 colums
stalcDF_N.drop(['Dalc', 'Walc'], axis=1, inplace=True) #drop the 3 colums

#drop any text/duplciated columns 

stalcDF_Numeric= stalcDF_N.copy() 
print("Length of stalcDF_N before dropping textual columns:",len(stalcDF_Numeric.columns)) #should print 47 on the first run
for i in categorical_cols:
    if i in stalcDF_Numeric:
        #print("dropping", i, "\n....")
        stalcDF_Numeric.drop([i], axis=1, inplace=True)         #print("dropped", i, "!")
    else:
        pass
print("\nLength of stalcDF_N after dropping textual columns:",len(stalcDF_Numeric.columns)) 



#CREATE THE FEATURE SET FIRST. = dataset.data

stalcDF_dataset_data_df = stalcDF_Numeric.copy() #i dont want to edit the original 
stalcDF_dataset_data_df.drop(["sex_CAT"], axis=1, inplace=True)#could also drop the highlighy correlated attributes too but will hold off for now
stalcDF_dataset_data = stalcDF_dataset_data_df.to_numpy()


#create "dataset.feature_names" 
stalcDF_feature_names = str(set(list(stalcDF_dataset_data_df.columns))) #<class 'str'>
    

  #need numerics for some alogo's so swap to a numeric df
df = stalcDF_Numeric.copy()
X = df.drop(["sex_CAT"], axis = "columns") #features
y = df.sex_CAT                             #label  



#**** The money maker!!!!! ****# 



model_params = {

#use the following two lines to check what params can be edited for each classifier
    # x = GaussianNB()
    # print(x.get_params().keys())   
    
    "svm": {
        "model": svm.SVC(gamma="auto"),
        "params": {
            "C":[2,3,5],
            "kernel":["rbf","linear"]
        }
        
    },
    "random_forest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators":[1,4,5]
        }
    },
    "logistic_regression": {
        "model": LogisticRegression(solver="liblinear",multi_class="auto"),
        "params": {
            "C":[1,4,5]
        }
    },
    "decision_tree": {
        "model": DecisionTreeClassifier(criterion="gini"),
        "params": {
            "max_depth":[1,2,3,4,5,6,7]
        }
    },
    "Gaussian_Bayes": {
        "model": GaussianNB(),
        "params": {
            "var_smoothing":[2.5,2.75,3,3.25,3.5] 
        }
    },
    "kNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors":[3,5,7,9,11,13],
            "weights": ['uniform', 'distance'],
            "leaf_size" :[3,4,5,6,7,8,9]
        }
    }
}
                                     
scores = []
for model_name, mp in model_params.items():
    clf = GridSearchCV(mp["model"],mp["params"],cv=5,return_train_score=False)
    clf.fit(X,y)
    scores.append({
        "model": model_name,
        "best_score": clf.best_score_,
        "best_params": clf.best_params_
    })
df = pd.DataFrame(scores,columns = ["model","best_score","best_params"])   
df

