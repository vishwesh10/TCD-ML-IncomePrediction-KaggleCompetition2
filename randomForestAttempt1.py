#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:09:01 2019
@author: abishekvaithylingam
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
#from sklearn.svm import SVC
#from ensembler import blend
from sklearn.ensemble import ExtraTreesRegressor
#from catboost import CatBoostRegressor
#from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
#from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
#from sklearn.metrics import mean_squared_error, mean_absolute_error
#from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
#from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from mlxtend.classifier import StackingClassifier
#from math import sqrt, log, exp





def managingNulls(dataFrame):
    #Housing Situation [dataType = category] -> No Hair 
    dataFrame['Work Experience in Current Job [years]'] = dataFrame['Work Experience in Current Job [years]'].fillna(dataFrame['Work Experience in Current Job [years]'].mean())
    
    #'Satisfaction with employer' -> Unknown
    dataFrame['Satisfation with employer'] = dataFrame['Satisfation with employer'].fillna('Unknown')
    
    dataFrame['Hair Color'] = dataFrame['Hair Color'].fillna('No Hair')
    
    #currentYear = float(datetime.datetime.now().year)
    dataFrame['Year of Record'] = dataFrame['Year of Record'].fillna(dataFrame['Year of Record'].median())

    #Gender [dataType = object] -> Unknown Gender
    dataFrame['Gender'] = dataFrame['Gender'].fillna('Unknown Gender')

    #Age [dataType = float64] -> mean
    dataFrame['Age'] = dataFrame['Age'].fillna(dataFrame['Age'].mean())

    #Profession [dataType = object] -> No Profession
    dataFrame['Profession'] = dataFrame['Profession'].fillna('No Profession')

    #University Degree [dataType = object] -> No Degree
    dataFrame['University Degree'] = dataFrame['University Degree'].fillna('No Degree')

    #Hair Color [dataType = object] -> No Hair 
    dataFrame['Hair Color'] = dataFrame['Hair Color'].fillna('No Hair')

    return dataFrame
                                                               




def FormattingColumn(dataFrame):
    
    #print(dataFrame.dtypes())
    
    #Housing situation -> ['0'] -> Unknown
    dataFrame['Housing Situation'] = dataFrame['Housing Situation'].replace(['0',0.0,'nA'],'Unknown')
    #print(dataFrame['Housing Situation'].unique())
    
    #Work Experience in Current Job [years] -> [#NUM!] -> mean
    dataFrame['Work Experience in Current Job [years]'] = dataFrame['Work Experience in Current Job [years]'].replace(['#NUM!',0.0],0)

    
    #Gender => ['0','unknown'] -> Unknown Gender | ['other'] -> Other Gender
    dataFrame['Gender'] = dataFrame['Gender'].replace(['0','unknown'],'Unknown Gender')
    dataFrame['Gender'] = dataFrame['Gender'].replace(['other'],'Other Gender')
    #dataFrame['Gender'] = pd.to_numeric(dataFrame['Gender'])
    
    #University Degree => ['No','0'] -> No Degree 
    dataFrame['University Degree'] = dataFrame['University Degree'].replace(['No','0'],'No Degree')

    #Hair Color => ['Unknown','0'] -> Unknown Hair Color
    dataFrame['Hair Color'] = dataFrame['Hair Color'].replace(['Unknown','0'],'Unknown Hair Color')
    
    #Country => ['0'] -> Unknown
    dataFrame['Country'] = dataFrame['Country'].replace(['0',0.0],'Unknown')
    
    #Yearly Income in addition to Salary (e.g. Rental Income) => Convert to numeric column
    dataFrame['Yearly Income in addition to Salary (e.g. Rental Income)'] = dataFrame['Yearly Income in addition to Salary (e.g. Rental Income)'].str.replace("EUR","")
    dataFrame['Yearly Income in addition to Salary (e.g. Rental Income)'] = pd.to_numeric(dataFrame['Yearly Income in addition to Salary (e.g. Rental Income)'])

    return dataFrame





def ScalingColumns(dataFrame,columnName):
    scaler = preprocessing.MinMaxScaler()
    column = dataFrame[columnName].astype(float)
    column = np.array(column).reshape(-1,1)
    scaledColumn = scaler.fit_transform(column)
    normalizeColumn = pd.DataFrame(scaledColumn)
    dataFrame[columnName] = normalizeColumn
    return dataFrame





def oneHotEncoder(uniqueFeautes,dataFrame,columnName):
    
    # OneHeartEncoder
    encoder = OneHotEncoder(categories = [uniqueFeautes],sparse = False, handle_unknown = 'ignore')

    #reshape the column
    column = dataFrame[columnName]
    column = np.array(column).reshape(-1,1)

    #Extract the column and join the data frame
    dataFrame = dataFrame.join(pd.DataFrame(encoder.fit_transform(column),columns=encoder.categories_,index=dataFrame.index))

    return dataFrame





def add_noise(series, noise_level):
    #Fnction to add noise to the series
    return series * (1 + noise_level * np.random.randn(len(series)))





def targetEncoder(trn_series,tst_series,target):
    #Function to preform Target encoding
    min_samples_leaf=1 
    smoothing=1,
    noise_level=0
    temp = pd.concat([trn_series, target], axis=1)

    #Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])

    #Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))

    #Apply average function to all target data
    prior = target.mean()

    #The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)

    # Apply average
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)

    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])

    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))

    # Apply average function to all target data
    prior = target.mean()

    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)

    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index

    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)





def preprocessingFormattingDS(training_dataset,test_dataset):
    #Function to pre process the dataframe
    
    #Check for duplicate records
    #training_dataset[training_dataset['Instance'].duplicated()]
    #test_dataset[test_dataset['Instance'].duplicated()]
    
    #Removing Instance Column
    training_dataset = training_dataset.loc[:, [
    'Year of Record', 'Housing Situation', 'Crime Level in the City of Employement','Work Experience in Current Job [years]','Satisfation with employer', 'Gender', 'Age', 'Country', 'Size of City', 'Profession', 'University Degree', 'Wears Glasses', 'Hair Color', 'Body Height [cm]', 'Yearly Income in addition to Salary (e.g. Rental Income)']]
    test_dataset = test_dataset.loc[:, [
    'Year of Record', 'Housing Situation', 'Crime Level in the City of Employement','Work Experience in Current Job [years]','Satisfation with employer', 'Gender', 'Age', 'Country', 'Size of City', 'Profession', 'University Degree', 'Wears Glasses', 'Hair Color', 'Body Height [cm]', 'Yearly Income in addition to Salary (e.g. Rental Income)']]
    
    #Formatting columns
    FormattingColumn(training_dataset)
    FormattingColumn(test_dataset)
    
    #Managing NaNs
    managingNulls(training_dataset)
    managingNulls(test_dataset)
    
    return training_dataset,test_dataset





def rfr_model(X, y):
    rfr = RandomForestRegressor(n_estimators=500,n_jobs=-1)
    
    return rfr





def model_fit(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    return model





def gbr_model(X,y):
    gbr = GradientBoostingRegressor(n_estimators=500)
    #gbr.max_depth = 6
    return gbr




def rem_outliers(df):
    df_ = df.copy()
    median = df['Income in EUR'].median()
    std = df['Income in EUR'].std()
    df_ = df_[df_['Income in EUR'] >= median - (2*std)]
    df_ = df_[df_['Income in EUR'] <=  median + (2*std)]
    return df



'''
#Data that falls within Q1 - 1.5IQR & Q3 + 1.5IQR are retained.
def outlier_removal(feature):
    q3, q1 = np.percentile(df[feature], [75 ,25])
    iqr = q3 - q1
    ul = q3 + (1.5*iqr)
    ll = q1 - (1.5*iqr)
    not_outliers = 0
    outliers = 0
    for i in range(len(df)):
        if ll <= df.loc[i, feature] <= ul:
            not_outliers += 1
        else:
            df.drop(index=i, inplace=True)
            outliers +=1
    print('Not outliers: ', not_outliers)
    print('Outliers: ', outliers)
    return
'''




def outlier_removal(df, feature):
    '''removes outliers from features passed to the function and returns a dataframe.'''
    q3, q1 = np.percentile(df[feature], [75 ,25])
    iqr = q3 - q1
    ul = q3 + (1.5*iqr)
    ll = q1 - (1.5*iqr)
    not_outliers = 0
    outliers = 0
    for i in range(len(df)):
        if ll <= df.loc[i, feature] <= ul:
            not_outliers += 1
        else:
            df.drop(index=i, inplace=True)
            outliers +=1
        
    print('\t\tFor ', feature)
    print('\t\tNot outliers: ', not_outliers)
    print('\t\tOutliers: ', outliers, '\n')
    
    return df




def evaluateMAE(model,crossValScore):
    score = np.sqrt(-crossValScore)
    print('Cross validation score:\n')
    print('Local MAE: ', round(score.min(), 4))
    
    
    
    
    
def printRegressionStats(y_test, model):
    #Prints regression evaluation metrics.
    y_test = y_test.reshape(-1, 1)
    print('Length of testing data: ', len(y_test))
    
    predictions = np.exp(model.predict(y_test))
    print('Local MAE: ', np.round(mean_absolute_error(y_test, predictions), 4))
    #print('RMSE on original price:', round(np.sqrt(mean_squared_error(np.expm1(y_test), model_pred)), 4))
   # print('RMSE on log transformed price:', round(np.sqrt(mean_squared_error(y_test, np.log1p(model_pred))), 4))

#     print('MSE: ', np.round(mean_squared_error(y_test, model_pred), 4))
#     print('MAE: ', np.round(mean_absolute_error(y_test, model_pred), 4))
    print('\n')




def run():
    training_dataset = pd.read_csv('tcd-ml-1920-group-income-train.csv',na_values = ['#NUM!'])
    test_dataset = pd.read_csv('tcd-ml-1920-group-income-test.csv',na_values = ['#NUM!'])
    
    #training_dataset.describe()
    #Drop the 'Total Yearly Income [EUR]' column from test_dataset
    test_dataset.drop(['Total Yearly Income [EUR]'], axis = 1)
                                                                             
    y_train = training_dataset['Total Yearly Income [EUR]']
    
    #Counting the number of numeric and categorical columns
    numeric_columns = training_dataset.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = training_dataset.select_dtypes(include=np.object).columns.tolist()
    
    #Print the number of numeric and categorical columsn
    print('\nNumerical columns: ', len(numeric_columns),'\n', numeric_columns, '\n')
    print('Categorical columns: ', len(categorical_columns), '\n', categorical_columns, '\n')
    
    #Features with their corresponding missing values.
    #print('\nNumber of columns in Training Dataset with missing values: ', len(training_dataset.isnull().sum()[training_dataset.isnull().sum() > 0]))
    #print(training_dataset.isnull().sum()[training_dataset.isnull().sum() > 0])
    
    #Removing columns with more than 25% of missing values
    threshold = len(training_dataset) * 0.25
    drop_cols = training_dataset.isnull().sum()[training_dataset.isnull().sum() > threshold].index.values.tolist()
    training_dataset.drop(columns=drop_cols, inplace=True)
    print('\nNumber of columns dropped: ', len(drop_cols))
    
    #Preprocessing the data 
    training_dataset,test_dataset = preprocessingFormattingDS(training_dataset,test_dataset)
    
    
    #Encoding CATEGORICAL DATA
    # => Housing Situation
    #oneHotEncoder(uniqueHousingSituation,training_dataset,'Housing Situation')
    #oneHotEncoder(uniqueHousingSituation,test_dataset,'Housing Situation')
    training_dataset['Housing Situation'],test_dataset['Housing Situation']= targetEncoder(training_dataset['Housing Situation'],test_dataset['Housing Situation'],y_train)
    
    # => Satisfation with employer
    #oneHotEncoder(uniqueSatisfactionWithEmployer,training_dataset,'Satisfation with employer')
    #oneHotEncoder(uniqueSatisfactionWithEmployer,test_dataset,'Satisfation with employer')
    training_dataset['Satisfation with employer'],test_dataset['Satisfation with employer']= targetEncoder(training_dataset['Satisfation with employer'],test_dataset['Satisfation with employer'],y_train)
    
    # => Gender
    #oneHotEncoder(uniqueGender,training_dataset,'Gender')
    #oneHotEncoder(uniqueGender,test_dataset,'Gender')
    training_dataset['Gender'],test_dataset['Gender']= targetEncoder(training_dataset['Gender'],test_dataset['Gender'],y_train)

    # => Country
    training_dataset['Country'],test_dataset['Country']= targetEncoder(training_dataset['Country'],test_dataset['Country'],y_train)
    
    # => Profession
    training_dataset['Profession'],test_dataset['Profession']= targetEncoder(training_dataset['Profession'],test_dataset['Profession'],y_train)
    
    # => University
    #oneHotEncoder(uniqueUniversity,training_dataset,'University Degree')
    #oneHotEncoder(uniqueUniversity,test_dataset,'University Degree')
    training_dataset['University Degree'],test_dataset['University Degree']= targetEncoder(training_dataset['University Degree'],test_dataset['University Degree'],y_train)

    # => Hair
    training_dataset['Hair Color'],test_dataset['Hair Color']= targetEncoder(training_dataset['Hair Color'],test_dataset['Hair Color'],y_train)
    
    '''    
    #Handling Outliers
    columns = training_dataset.columns.tolist()
    for i in columns:
        outlier_removal(training_dataset,i)
    '''
    
    #Understanding the features and their datatypes
    #plt.figure(figsize=(24,20))
    #cor = training_dataset.corr()
    #sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    #plt.show()
    
    #X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(training_dataset, y_train, test_size=0.3, random_state=42)
    #kf = KFold(n_splits=5)
    
    #Take log of all the values in the y_train column
    y_train = np.log(y_train)
    
    '''
    regressorModels = [
          RandomForestRegressor(n_estimators=5,n_jobs=-1),
          GradientBoostingRegressor(n_estimators=5),
          ExtraTreesRegressor(n_estimators=10, max_depth=6, max_features=None, n_jobs=-1),
          ]
    =
    #Creating Ensemble Regressor model
    print("Creating Ensemble Regressor model...")
    ensembleModel = blend.BlendModel(regressorModels, nFoldsBase=3)
    print("Created Ensemble Regressor model.")
    
    #Fitting Ensemble Regressor model
    print("Fitting Ensemble Regressor model...")
    ensembleModel.fit(training_dataset, y_train)
    print("Done")
    
    predictions = ensembleModel.predict(test_dataset)
    
    '''
    #Create Random Forest Regressor model
    print("Creating Random Forest Regressor model...")
    randomForestModel = rfr_model(training_dataset,y_train)
    print("Created Random Forest Regression model.")
    
    #Create Gradient Boost Regressor model
    print("Creating Gradient Boost Regressor model...")
    gradientBoostModel = gbr_model(training_dataset,y_train)
    print("Created Gradient Boost Regressor model.")
    
    
    
    #crossValScore = cross_val_score(randomForestModel, X_train_val, y_train_val, cv=kf, scoring='neg_mean_absolute_error')
    #evaluateMAE(randomForestModel,crossValScore)
    
    
    #Fit the Random Forest Regressor model
    print("Fitting the Random Forest Regressor model...")
    randomForestModel.fit(training_dataset,y_train)
    print("Completed Random Forest Regressor model fitting")
    
    #Fit the Gradient Boost Regressor model
    print("Fitting the Gradient Boost Regressor model...")
    gradientBoostModel.fit(training_dataset,y_train)
    print("Completed Gradient Boost Regressor model fitting")
    
    
    #Making predictions
    print("Starting predictions with RFR...")
    predictionsRFR = np.exp(randomForestModel.predict(test_dataset))
    print("Completed predictions")
    
    print("Starting predictions with GBR...")
    predictionsGBR = np.exp(gradientBoostModel.predict(test_dataset))
    print("Completed predictions")
    
    '''
    print("Fitting Local Test model")
    randomForestModel = model_fit(randomForestModel, X_train_val, y_train_val, X_test_val)
    print("Completed fitting local test model")
    print(randomForestModel)
    printRegressionStats(y_test_val, randomForestModel)
    '''
    
    print("Saving RFR predictions to a file") 
    np.savetxt('predictedRFROutput.csv',predictionsRFR)
    
    print("Saving GBR predictions to a file") 
    np.savetxt('predictedGBROutput.csv',predictionsGBR)
    
    return



if __name__ == '__main__':
    run()
    
    
    
    
    
    
    
    
    
    
    
    
    
    #List of Categorical Data columns and encoding strategies for each 
    # -> Housing Situation - TARGET
    # -> Satisfation with employer - TARGET
    # -> Gender - TARGET
    # -> Country - TARGET
    # -> Profession - TARGET
    # -> University Degree - TARGET
    # -> Hair Color - TARGET
    
    
    #Extracting UNIQUE features from Profession and Country columns
    #uniqueSatisfactionWithEmployer = training_dataset['Satisfation with employer'].unique()
    #uniqueGender = training_dataset['Gender'].unique()
    #uniqueUniversity = training_dataset['University Degree'].unique()
    #uniqueHousingSituation = training_dataset['Housing Situation'].unique()    
    

'''
    
    #Understanding the features and their datatypes
    plt.figure(figsize=(12,10))
    cor = training_dataset.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()
    
    print(training_dataset.describe())
    print("\nYear of Record")
    print(training_dataset['Year of Record'].unique())
    print("\nHousing Situation")
    print(training_dataset['Housing Situation'].unique())
    print("\nCrime Level in the City of Employement")
    print(training_dataset['Crime Level in the City of Employement'].unique())
    print("\nWork Experience in Current Job [years]")
    print(training_dataset['Work Experience in Current Job [years]'].unique())
    print("\nSatisfation with employer")
    print(training_dataset['Satisfation with employer'].unique())
    print("\nGender")
    print(training_dataset['Gender'].unique())
    print("\nAge")
    print(training_dataset['Age'].unique())
    print("\nCountry")
    print(training_dataset['Country'].unique())
    print("\nSize of City")
    print(training_dataset['Size of City'].unique())
    print("Profession")
    print(training_dataset['Profession'].unique())
    print("\nUniversity Degree")
    print(training_dataset['University Degree'].unique())
    print("\nWears Glasses")
    print(training_dataset['Wears Glasses'].unique())
    print("\nHair Color")
    print(training_dataset['Hair Color'].unique())
    print("\nBody Height [cm]")
    print(training_dataset['Body Height [cm]'].unique())
    print("\nYearly Income in addition to Salary (e.g. Rental Income)")
    print(training_dataset['Yearly Income in addition to Salary (e.g. Rental Income)'].unique())
    print("Total yearly income")
    print(training_dataset['Total Yearly Income [EUR]'].unique())

'''
        
        
