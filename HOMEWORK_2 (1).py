#!/usr/bin/env python
# coding: utf-8

# In[141]:


get_ipython().system('pip install xgboost')



# In[142]:



#In this project, you are tasked to build a model(s) that predict a player's overall rating given the player's profile.
#Demonstrate the data preparation & feature extraction process
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt


# In[103]:


import pandas as pd
import numpy as np
file = "male_players (legacy).csv"
file22="players_22.csv"
data22 = pd.read_csv(file22)
# Read the CSV file
data = pd.read_csv(file)


# In[104]:


# Step 2: Drop specified columns
columns_to_drop = [
    'player_id', 'dob', 'player_tags', 'club_contract_valid_until_year', 'player_url', 'club_jersey_number', 'club_name', 'fifa_update',
    'long_name', 'short_name', 'league_id', 'player_face_url', 'nationality_id', 'preferred_foot', 'club_contract_valid_until_year',
    'fifa_update_date', 'club_position', 'league_name', 'club_team_id', 'nation_team_id', 'player_traits', 'club_joined_date', 'league_level',
    'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 'ldm', 'cdm', 'rdm', 'lb', 'cb', 'nationality_name', 'real_face',
    'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lwb', 'rwb', 'lcb', 'rcb', 'rb', 'gk', 'player_face_url', 'ls', 'st', 'body_type', 'fifa_version'
]
columns_to_drop22 = [
    'sofifa_id', 'dob', 'player_tags', 'club_contract_valid_until', 'player_url', 'club_jersey_number', 'club_name','club_loaned_from','nation_position',
    'long_name', 'short_name', 'player_face_url', 'nationality_id', 'preferred_foot','club_position', 'league_name', 'club_team_id', 'nation_team_id', 'player_traits', 'club_joined', 'league_level',
    'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 'ldm', 'cdm', 'rdm', 'lb', 'cb', 'nationality_name', 'real_face','nation_logo_url','nation_flag_url',
    'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lwb', 'rwb', 'lcb', 'rcb', 'rb', 'gk', 'player_face_url', 'ls', 'st', 'body_type','club_logo_url','club_flag_url'
]
data = data.drop(columns=columns_to_drop, axis=1)
data22=data22.drop(columns=columns_to_drop22, axis=1)

# Step 3: Drop columns with 30% or more null values
biased_threshold = 0.30 * len(data)
data = data.loc[:, data.isna().sum() < biased_threshold]
data22=data22.loc[:, data22.isna().sum() < biased_threshold]


# In[105]:


# Function to check for mixed data types in a column
def check_mixed_types(column):
    types = column.apply(lambda x: type(x)).unique()
    return len(types) > 1

#  Separate numeric data from non-numeric data
numeric_data = data.select_dtypes(include=np.number)
numeric_data22=data22.select_dtypes(include=np.number)
non_numeric_data = data.select_dtypes(include=['object'])
non_numeric_data22 = data22.select_dtypes(include=['object'])

#  Impute missing values for numeric data with mean
numeric_data = numeric_data.apply(lambda col: col.fillna(col.mean()))
numeric_data22 = numeric_data22.apply(lambda col: col.fillna(col.mean()))

# Impute missing values for non-numeric data with mode
for column in non_numeric_data.columns:
    mode = non_numeric_data[column].mode()[0]
    non_numeric_data[column].fillna(mode, inplace=True)

# Impute missing values for non-numeric22 data with mode
for column in non_numeric_data22.columns:
    mode = non_numeric_data22[column].mode()[0]
    non_numeric_data22[column].fillna(mode, inplace=True)















# In[106]:


# Encode categorical columns using LabelEncoder
label_encoders = {}
for column in non_numeric_data.columns:
    label_encoders[column] = LabelEncoder()
    non_numeric_data[column] = label_encoders[column].fit_transform(non_numeric_data[column])

# Encode categorical columns for test data using the same LabelEncoders
for column in non_numeric_data22.columns:
    if column in label_encoders:
        non_numeric_data22[column] = label_encoders[column].transform(non_numeric_data22[column])
    else:
        # Handle cases where the test data might have columns not present in the training data
        print(f"Warning: Column {column} not found in training data encoders and will be ignored.")
        non_numeric_data22 = non_numeric_data22.drop(columns=[column])
    # Combine numeric data and label encoded data
processed_data = pd.concat([numeric_data, non_numeric_data], axis=1)
data22=pd.concat([numeric_data22, non_numeric_data22], axis=1)
corr_matrix=processed_data.corr()


# In[138]:


definers = []
for col in processed_data.columns:
    if col != 'overall':
        if (corr_matrix.loc['overall', col] > 0.45) or (corr_matrix.loc['overall', col] < -0.45):
            definers.append(col)
correlation_dict = {col: corr_matrix.loc['overall', col] for col in definers}

# Sort the definers list based on the correlation values with 'overall' in descending order
sorted_definers = sorted(definers, key=lambda x: correlation_dict[x], reverse=True)
sorted_definers


# In[139]:


processed_data


# In[140]:


# scaling te data after label encoding
overall=processed_data['overall']
processed_data=processed_data[sorted_definers]
scaler=StandardScaler()
scaler_2=scaler.fit_transform(processed_data)
scaler_2
processed_data=pd.DataFrame(scaler_2,columns=processed_data.columns)


# TRAINING AND TESTING

# In[110]:


# splitting data for train and test.
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(processed_data,test_size=0.2, random_state=42)  # splitt into two sets using 20 % 80%
print(len(train_set), "train +", len(test_set), "test")  # splitting datset using 20 % 80%


# In[111]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X = processed_data.copy()
y=overall


# 

# In[112]:


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# getting best features
forest=RandomForestRegressor(n_estimators=250,n_jobs=-1)
forest=forest.fit(Xtrain, Ytrain)
forest_predictions = forest.predict(Xtest)
forest_score=forest.score(Xtest,Ytest)
forest_rmse = np.sqrt(mean_squared_error(Ytest, forest_predictions))

print(f'RandomForestRegressor RMSE: {forest_rmse}')

print(f'test score: {forest_score}')



# In[113]:


# Initialize and fit GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=250, random_state=42)
gbr=gbr.fit(Xtrain, Ytrain)
gbr_predictions = gbr.predict(Xtest)
gbr_rmse = np.sqrt(mean_squared_error(Ytest, gbr_predictions))
gbr_score=gbr.score(Xtest,Ytest)
print(f'GradientBoostingRegressor RMSE: {gbr_rmse}')
print(f'test_score: {gbr_score}')


# In[114]:


xgb = XGBRegressor(n_estimators=250, random_state=42, n_jobs=-1)
xgb.fit(Xtrain, Ytrain)
xgb_predictions = xgb.predict(Xtest)
xgb_rmse = np.sqrt(mean_squared_error(Ytest, xgb_predictions))
xgb_score=xgb.score(Xtest,Ytest)
print(f'XGBoostRegressor RMSE: {xgb_rmse}')
print(f'test_score: {xgb_score}')


# UTILIZING FEATURE IMPORTANCE

# In[115]:


# Get feature importances for the forest regressor because it was the best model
feature_importances = forest.feature_importances_

# Get feature names
feature_names = Xtrain.columns

# Create a dictionary of feature names and their importances
feature_importance_dict = dict(zip(feature_names, feature_importances))

# Sort features by importance (from highest to lowest)
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# Select the top 7 features based on importance
selected_features = sorted_features[:7]

# Extract the names of the selected features
selected_feature_names = [feature[0] for feature in selected_features]


# TESTING THE NEW IMPORTANT FEATURES

# In[116]:


# Utilize the selected features
X_train_selected = Xtrain[selected_feature_names]
X_test_selected = Xtest[selected_feature_names]

# Initializing and train a new model with selected features
forest = RandomForestRegressor(n_estimators=250, random_state=42)
forest.fit(X_train_selected, Ytrain)

# Make predictions with the new model
y_pred_selected = forest.predict(X_test_selected)

# Evaluate the new model's performance
from sklearn.metrics import mean_squared_error
rmse_selected = np.sqrt(mean_squared_error(Ytest, y_pred_selected))
print(rmse_selected)


# CROSS-VALIDATION AND FINETUNING
# 

# In[117]:


#selectin random forest because it gave te least rmse and highest test score.
rf_model = {
    'name': 'RandomForestRegressor',
    'model': RandomForestRegressor(),
    'params': {
        'n_estimators': [50, 100, 150],
        'max_depth': [ 10, 20,30],
    }
}


# In[122]:


#saving a scaler model for the flask deployment
import pickle
model=StandardScaler()
filename=r'C:\Users\hp\Documents\TEXT BOOKS\year 3_sem2\jupyternotebooks\scaler.pkl' 
with open(filename, 'wb') as file:
    pickle.dump(model,file)


# In[123]:


from sklearn.metrics import make_scorer, mean_squared_error


# In[124]:




# Function to perform GridSearchCV and return best model and results
def perform_grid_search(model_config, Xtrain, Ytrain):
    grid_search = GridSearchCV(model_config['model'], model_config['params'], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(Xtrain, Ytrain)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_estimator = grid_search.best_estimator_

    return best_params, best_score, best_estimator

# Assuming Xtrain and Ytrain are already defined

# Perform GridSearchCV for RandomForestRegressor
best_params, best_score, best_rf_model = perform_grid_search(rf_model, Xtrain, Ytrain)

# Print the results
print(f"Results for {rf_model['name']}:\n")
print(f"Best Parameters: {best_params}")
print(f"Best  MSE: {best_score}")
print("=============================================")


# SAVING THE MODEL
# 

# In[125]:


import pickle


filename=r'C:\Users\hp\Documents\TEXT BOOKS\year 3_sem2\jupyternotebooks\best_rf_model.pkl' 
with open(filename, 'wb') as file:
    pickle.dump(best_rf_model,file)


# In[134]:


overall=data22['overall']
y_22=data22['overall']
X_22=data22.drop(overall)
scaler=StandardScaler()
x_22=scaler.fit_transform(X_22)
x_22


# In[132]:


# load the model from disk to test
best_rf_model = pickle.load(open(filename, 'rb'))
y_new_pred = best_rf_model.predict(X_22[sorted_definers])



# EVALUATING MODEL PERFORMANCE ON NEW DATASET

# In[130]:


new_mse = mean_squared_error(y_22[:-47], y_new_pred)
print(f'Mean Squared Error on New Data: {new_mse}')      # te code is wrong


# In[ ]:




