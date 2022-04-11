'''Aggregating lat/lon data into a single predictive feature.'''

# %%
# dependencies
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer

# %%
# make a regression example
x,y = make_regression(n_samples=1000,
                      n_features=1,
                      n_informative=1,
                      n_targets=1,
                      noise=20,
                      bias=0)

# plot the data
fig,ax = plt.subplots(dpi=100)
ax.scatter(x,y, alpha=.5)
ax.set_title('How X Relates to Y')
ax.set_ylabel('Target (Y)')
ax.set_xlabel('Predictor (X)')
# %%
# import california housing data
caHousing = fetch_california_housing(as_frame=True)
# assign the dataset to a variable
df = caHousing['data']
df['target'] = caHousing['target']
# %%
# show some scatterplots
fig,(ax1,ax2) = plt.subplots(1,2,dpi=100, figsize=[6,3])
ax1.scatter(df['Latitude'],df['target'], alpha=.2)
ax2.scatter(df['Latitude'],df['target'], alpha=.2)
ax1.set_ylabel('$/100,000');
ax1.set_xlabel('Latitude');ax2.set_xlabel('Longitude') 


# %%
# show the dat
# sns.kdeplot(data=df, x='Longitude', y='Latitude', fill=True)
# %%
# px.scatter_geo(data_frame=df, 
#                lat='Latitude', 
#                lon='Longitude', 
#                color='target',
#                center={'lat':36.5,'lon':-119.5},
#                scope='usa')
# %%
# set features in the model
features = ['Latitude', 'Longitude']

# train test split
xTrain, xTest, yTrain, yTest = train_test_split(df[features], 
                                                df['target'],
                                                test_size=.3)

rmse = make_scorer(mean_squared_error, squared=False)

def objective(trial):
    '''
    Objective function for study to call when searching for 
    optimal parameters.
    '''
    # pick some random forest parameters to search over
    max_depth = trial.suggest_int('max_depth',
                                  2,
                                  40,
                                  log=True)
    min_samples_leaf = trial.suggest_int('min_samples_leaf',
                                         10,
                                         100,
                                         log=True)

    # instantiate a random forest regressor with params from optuna
    rfr = RandomForestRegressor(max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf,
                                n_jobs=-1)

    # score the model
    score = cross_val_score(rfr, 
                            xTrain, 
                            yTrain, 
                            n_jobs=4, 
                            cv=5,
                            scoring=rmse)
    
    return score.mean()

# create a study object and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# %%
# instantiate a random forest regressor with final params
rfr = RandomForestRegressor(max_depth=study.best_params['max_depth'],
                            min_samples_leaf=study.best_params['min_samples_leaf'],
                            n_jobs=-1)
# fit to all training data
rfr.fit(xTrain, yTrain)
# %%
# see how it performs on test data
resultsdf = pd.DataFrame(yTest).copy()
resultsdf['yPred'] = rfr.predict(xTest)
resultsdf.sort_values('target', inplace=True)
resultsdf.reset_index(inplace=True)
# %%
# plot the results
fig, ax = plt.subplots(dpi=100)
ax.scatter(resultsdf.index, resultsdf['yPred'], label='Random Forest Prediction',c='C1',alpha=.3)
ax.scatter(resultsdf.index, resultsdf['target'], label='Truth', c='C0')
ax.set_xlabel('Test Data Records Sorted by True District Home Value')
ax.set_ylabel('$ / 100,000')
ax.legend()
ax.set_title('CA District True and Predicted Home Values')
# %%
fig, ax = plt.subplots(dpi=100)
ax.hist(resultsdf['target'] - resultsdf['yPred'], bins=50)
ax.set_xlabel('Difference between Truth and Prediction')
ax.set_ylabel('Count of Districts')
ax.set_title('CA District True Minus Predicted Home Values')
# %%
# rolling window mean

def rollingWindowMean(df, feature, winSize=6000, target='target'):

    df.sort_values(feature, inplace=True)
    rollData = df.rolling(winSize, 
                          min_periods=int(winSize/2), 
                          center=True, 
                          win_type='triang',
                          closed='both').mean()

    fig, ax = plt.subplots(dpi=100)
    ax.scatter( rollData[feature],rollData['target'])
    plt.show()
# %%

rollingWindowMean(df, 'Latitude')
# %%
rollingWindowMean(df, 'Longitude')
# %%
