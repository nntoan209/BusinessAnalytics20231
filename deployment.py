import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import os # accessing directory structure
from sklearn.metrics import mean_squared_error

import plotly.graph_objects as go
import plotly.express as px

import seaborn as sns
import matplotlib.pyplot as plotCorrelationMatrix

from sklearn import datasets, ensemble

from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from sklearn.inspection import permutation_importance

#-------------
#theme 
#--

primaryColor="#F63366"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"

#-------------
#layout design 
#-------------

st.title(' LIFE EXPECTANCY ESTIMATOR TOOL ')
st.markdown("---")
st.write('''
         This app will apply Gradient Boosting Regressor (GBM) Machine Learning algorithms to estimate the life expectancy based on 
features & data obtained from World Bank Open Data.
         
Please fill in the attributes below and adjust the model's parameters to the desired values.

Once ready, please hit the 'ESTIMATE LIFE EXPECTANCY' button to get the prediction and the GBM model's performance. 
''')
st.markdown("---")

#---------------------------------#
# Model building
st.header('Input Attributes')
att_male = st.slider('Male population', min_value=0, max_value=1000000000, value= 80, step=10)
att_female = st.slider('Female population', min_value= 0, max_value=1000000000, value=69, step=10)
att_pop_dense= st.slider('Population density', min_value= 0, max_value=300, step=0.5)
att_infant= st.slider('Infant death per 1000', min_value= 0, max_value=200, step=0.5)
att_mor =  st.slider('Mortality rate between 15 and 60 years old per 1000', min_value= 0, max_value=1000, step=0.5)
att_gdp = st.slider('GDP in billion USD', min_value= 0, max_value=50000, step=0.5)
att_gdp_growth = st.slider('GDP growth in percentage', min_value= -20, max_value=20, step=0.1)
att_gdp_per = st.slider('GDP per capita', min_value= 0, max_value=200000, step=100)
att_infla = st.slider('inflation rate', min_value= -20, max_value=100, step=0.2)
att_water = st.slider('Safe water service in percent', min_value= 0, max_value=100, step=0.5)
att_sanitation = st.slider('Safe sanitation service in percent', min_value= 0, max_value=100, step=0.5)
att_health_gdp = st.slider("Health expenditure gdp ", min_value = 0.0, max_value = 125.0, step = 0.5) 
att_health_capita = st.slider("Health expenditure capita ", min_value = 0.0, max_value = 12000.0, step = 5.0)
att_status = st.selectbox("Country status",options=(1,2))

st.write('''
         * 1: Developing
         * 2: Developed
         '''
         )
if att_status == 1:
    att_status_1 = 1
    att_status_2 = 0
elif att_status == 2: 
    att_status_2 = 1
    att_status_1 = 0

att_regn = st.selectbox('Region', options=(1,2,3,4,5,6,7))
st.write('''
         * 1: East Asia & Pacific
         * 2: Europe & Central Asia
         * 3: Latin America & Caribbean
         * 4: Middle East & North Africa
         * 5: North America
         * 6: South Asia
         * 7: Sub-Saharan Africa
         '''
         )

if att_regn == 1:
    att_regn_1 = 1
    att_regn_2 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = 0
elif att_regn == 2: 
    att_regn_2 = 1
    att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = 0
elif att_regn == 3: 
    att_regn_3 = 1
    att_regn_1 = att_regn_2 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 =  0
elif att_regn == 4: 
    att_regn_4 = 1
    att_regn_1 = att_regn_3 = att_regn_2 = att_regn_5 = att_regn_6 = att_regn_7 =  0
elif att_regn == 5: 
    att_regn_5 = 1
    att_regn_1 = att_regn_3 = att_regn_4 = att_regn_2 = att_regn_6 = att_regn_7 =  0
elif att_regn == 6: 
    att_regn_6 = 1
    att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_2 = att_regn_7 = 0
else:
    att_regn_7 = 1
    att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_2  = 0


user_input = np.array([att_male , att_female, att_pop_dense, att_infant, att_mor, att_gdp, att_gdp_growth, att_gdp_per, att_infla, att_water, att_sanitation, att_health_gdp, att_health_capita, att_status_1, att_status_2,
                       att_regn_1, att_regn_2, att_regn_3,
                       att_regn_4, att_regn_5, att_regn_6, att_regn_7, 
                       ]).reshape(1,-1)


    # AN Sidebar - Specify parameter settings
with st.sidebar.header('Set Parameters'):
        split_size = st.sidebar.slider('Data split ratio (percentage for Training Set)', min_value=10, max_value=90, value= 20, step=10) #use
        learning_rate = st.sidebar.select_slider('Learning rate (trade-off with n_estimators)', options=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]) #done
        parameter_n_estimators = st.sidebar.slider('Number of estimators (number of trees)', 100, 500, 1000) #done
        parameter_max_depth = st.sidebar.slider('Max depth (maximum number of levels in each trees)', min_value=1, max_value=9, value= 3, step= 1) #done
        parameter_max_features = st.sidebar.select_slider('Max features (Max number of features to consider at each split)', options=['auto', 'sqrt' , 'log2']) #done
        parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', min_value=1, max_value=10, value= 2, step= 1) #done
        parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', min_value=1, max_value=8, value= 1, step= 1) #done
        #AN addition
        parameter_max_leaf_node = st.sidebar.slider('Grow trees with max_leaf_nodes in best-first fashion', min_value=8, max_value=32, value= 8, step= 2) #done

        parameter_subsample = st.sidebar.select_slider('Subsample (percentage of samples per tree) ', options=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]) #done
        parameter_random_state = st.sidebar.slider('random_state (Controls the random seed given to each Tree estimator at each boosting iteration)',  min_value=0, max_value=100, value=100, step= 10) #done
        parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['friedman_mse', 'mse','mae']) #done

#------
# Model
#------

#import dataset
def get_dataset():
    data= pd.read_csv('integrated_data_fillna.csv')
    #data = pd.read_csv('ML_data_SL.csv')
    return data

life_df = get_dataset()
df = life_df.copy()
st.markdown("---")

if st.button('ESTIMATE LIFE EXPECTANCY'):
    data = get_dataset()
    
    #fix column names
    data.columns = (['Country', 'location_code', 'country_code', 'Type', 'Year',
       'total_population', 'male_population', 'female_population',
       'population_density', 'life_expectancy_at_birth',
       'male_life_expectancy_at_birth', 'female_life_expectancy_at_birth',
       'infant_deaths_per_1000', 'mortality_between_15_60_per_1000', 'gdp',
       'gdp_growth', 'gdp_per_capita', 'inflation_change',
       'total_safe_water_service', 'total_safe_sanitation_service', 'health_expenditure_over_gdp', 'health_expenditure_per_capita','Region',
       'Subregion', 'Status'])
    data.drop(['Type'],axis = 1)
    data.drop(['Subregion'],axis = 1)
    data.drop(['location_code'],axis = 1)

    #Fix data types
    data.Country = data.Country.astype('category')
    data.country_code = data.country_code.astype('category')
    data.year = data.year.astype('category')
    data.Region = data.Region.astype('category')
    data.Status = data.Status.astype('category')

    
    #Region Transform
    #data_final = pd.concat([data,pd.get_dummies(data['region'], prefix='region')], axis=1).drop(['region'],axis=1)
    
    #Data Split
    y = data['life_expectancy_at_birth']
    X = data.drop(['life_expectancy','Country','country_code', 'year','male_life_expectancy_at_birth', 'female_life_expectancy_at_birth'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-split_size)/100, random_state=parameter_random_state)

    gbm_opt = GradientBoostingRegressor(
        learning_rate =learning_rate,#use
        n_estimators=parameter_n_estimators, #use
        random_state=parameter_random_state, #use
        max_depth=parameter_max_depth, #use
        max_features=parameter_max_features, #use
        subsample= parameter_subsample, #use
        criterion=parameter_criterion, #use
        min_samples_split=parameter_min_samples_split, #use
        min_samples_leaf=parameter_min_samples_leaf,#use
        #AN Additional
        max_leaf_nodes=parameter_max_leaf_node)
    
    #model training

    gbm_opt.fit(X_train,y_train)

        #making a prediction
    gbm_predictions = gbm_opt.predict(user_input) #user_input is taken from input attributes 
    gbm_score = gbm_opt.score(X_test,y_test) #R2 of the prediction from user input
    gbm_mse = mean_squared_error(y_test, gbm_opt.predict(X_test))
    gbm_rmse = gbm_mse**(1/2)

    gbm_mse_train = mean_squared_error(y_train, gbm_opt.predict(X_train))
    gbm_rmse_train = gbm_mse_train**(1/2)

    ##AN - New
    st.markdown('**Result - Prediction!**')
    st.write('Based on the user input the estimated Life Expectancy is: ')
    st.info((gbm_predictions))

    # st.write('Based on the user input the estimated Life Expectancy for this region is: ')
    # st.info((gbm_predictions))

    st.subheader('Model Performance')

    st.write('With an ($R^2$) score of: ', gbm_score)
    
    st.write('Error (MSE or MAE) for testing:')
    st.info(gbm_mse)
    st.write("The root mean squared error (RMSE) on test set: {:.4f}".format(gbm_rmse))

    st.write('Error (MSE or MAE) for training:')
    st.info(gbm_mse_train)
    st.write("The root mean squared error (RMSE) on train set: {:.4f}".format(gbm_rmse_train))

    # deviantion chart *AN New*
    #Running new test to measure performance and test Deviance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=13)

    params = {'n_estimators': parameter_n_estimators,
            'max_depth': parameter_max_depth,
            'min_samples_split': parameter_min_samples_split,
            'learning_rate': learning_rate,
            'loss': 'ls'}
    #getting new MSE
    reg = ensemble.GradientBoostingRegressor(**params)
    reg.fit(X_train, y_train)

    mse = mean_squared_error(y_test, reg.predict(X_test))

    # Plotting Deviance
    st.subheader('Model Deviation Between Test & Train')
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
    for i, y_pred in enumerate(reg.staged_predict(X_test)):
        test_score[i] = reg.loss_(y_test, y_pred)

    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title('Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, reg.train_score_, 'b-',
            label='Training Set Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
            label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    fig.tight_layout()
    st.pyplot(fig)

    #Feature of importance
    st.subheader('Model Feature of Importance & Data Distribution')
    feature_names = X.columns
    # Plotting Features of importance
    feature_importance = reg.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    fig2 = plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, feature_names[sorted_idx])
    plt.title('Feature Importance (MDI)')

    result = permutation_importance(reg, X_test, y_test, n_repeats=10,
                                    random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()
    plt.subplot(1, 2, 2)
    plt.boxplot(result.importances[sorted_idx].T,
                vert=False, labels=feature_names[sorted_idx])
    plt.title("Permutation Importance (test set)")
    fig.tight_layout()
    st.pyplot(fig2)

    
    st.markdown('**Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('**Variable details**:')
    st.write('X variable - Attributes')
    st.info(list(X.columns))
    st.write('Y variable - Prediction')
    st.info(y.name)
    ##AN 
    
    #show parameters
    st.subheader('Model Parameters')
    st.write(gbm_opt.get_params())




# display data
st.markdown("---")
with st.container():
    show_data = st.checkbox("See the raw data?")

    if show_data:
        df
st.markdown("---")
# Life Expectancy Data.csv has 2939 rows in reality, but we are only loading/previewing the first 1000 rows
#df1 = pd.read_csv('https://renzo-test1.s3.amazonaws.com/life_expectancy/ML_data_SL.csv')
df1 = pd.read_csv('integrated_data_fillna.csv')
#df1.dataframeName = 'https://renzo-test1.s3.amazonaws.com/life_expectancy/ML_data_SL.csv'
df1.dataframeName = 'ML_data.csv'
nRow, nCol = df1.shape
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()

#Building Correlation Matrix Model for data
st.subheader('Correlation between features')
fig3 = plt.figure()
sns.heatmap(df1.corr())
st.pyplot(fig3)


