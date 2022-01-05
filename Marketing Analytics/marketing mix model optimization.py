# Business Problem: Lower than expected return on marketing spent  
# KPI: ROAS
# Business Requirements: Budget Optimization to improve ROAS


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.io as pio
import seaborn as sns
import plotly.express as px
import datetime as dt
import joblib
import pickle




# Step 1.0 Data Understanding
# weekly dataset from DATE
# data wrangling
# keep all column names in lowercase, already parse the date dtype
# facebook_i = impressions
pd.set_option('display.max_column', None, 'display.expand_frame_repr', False)
simulated_media_df = pd.read_csv('data/de_simulated_data.csv', parse_dates=['DATE'])
simulated_media_df.columns = simulated_media_df.columns.str.lower()
print(simulated_media_df.head())



# Step 1.0.1 Visualize Time Series Data using plotly
# pd.melt
# Unpivot a DataFrame from wide to long format, optionally leaving identifiers set.
# This function is useful to massage a DataFrame into a format
# where one or more columns are identifier variables (id_vars),
# while all other columns, considered measured variables (value_vars),
# are “unpivoted” to the row axis, leaving just two non-identifier columns, ‘variable’ and ‘value’.
df = simulated_media_df.melt(id_vars='date')
pio.renderers.default = 'browser'
fig = px.line(df,
				x = 'date',
				y = 'value',
				color = 'variable',
				facet_col = 'variable',
				facet_col_wrap = 2,
				template = 'plotly_dark').update_yaxes(matches=None)

fig.show()

fig_t = px.histogram(simulated_media_df[['date','tv_s']],
				x = 'date',
				y = 'tv_s',
				color = 'tv_s',
				template = 'plotly_dark')

fig_t.show()

fig_t = px.histogram(simulated_media_df[['date','revenue']],
				x = 'date',
				y = 'revenue',
				color = 'revenue',
				template = 'plotly_dark')

fig_t.show()



# Step 1.0.2 Visualize Spending Patterns
# regex: $ Matches the end of the string or just before the newline at the end of the string
# convert to dataframe: to_frame; set axis/name to the dataframe
df_spent = simulated_media_df.filter(regex='_s$', axis=1).apply(lambda x: x.sum())\
                             .to_frame().reset_index()\
                             .set_axis(['media', 'spent_current'], axis=1)
print(df_spent)
#
# cat vs numeric: bar chart
fig_s = px.bar(df_spent,
				x = 'media',
				y = 'spent_current',
				color = 'media',
				template = 'plotly_dark')

fig_s.show()



# Step 1.1 Add / Remove Features for Modeling
# For seasonality in time series data, we need to remove the seasonality and extract the month only
# in case of falsely detecting the revenue correlated with the seasonaly spending
# eg: Summer for more TV ads Spending
# pd.assign(column name =...)
# using dt.month_name instead of dt.month to avoid numeric expression in months (will be considered as ordinal data)
# dt.month_name will turn cat. data and using dummy variables
ds_media = simulated_media_df.assign(date_month=lambda x: x['date'].dt.month_name())\
                             .drop(['date', 'facebook_i','search_clicks_p'], axis=1)
print(ds_media)



# Step 1.2 Plot Correlations in Seaborn
fig_cor = sns.heatmap(
                    ds_media.corr(),
                    annot = True,
                    linewidths=.3,
                    center=0,
                    square = True)

plt.show()



# Step 2.0 Modeling with adstock using simple Advertising adstock theory (No diminish return considered)
# Construct a Basic Adstock Model
def adstock(series, rate):
    tt = np.empty(len(series))
    tt[0] = series[0]

    for i in range(1, len(series)):
        tt[i] = series[i] + tt[i - 1] * rate

    tt_series = pd.Series(tt, index=series.index)
    tt_series.name = f'adstock_{series.name}'

    return tt_series


adstock_tv_s=adstock(ds_media.tv_s,0.5)
adstock_ooh_s=adstock(ds_media.ooh_s,0.5)
adstock_print_s=adstock(ds_media.print_s,0.5)
adstock_search_s=adstock(ds_media.search_s,0.5)
adstock_facebook_s=adstock(ds_media.facebook_s,0.5)



# Step 2.1 Machcine Learning Sklearn Adstock Model
# Step 2.1.1 Feature Engineering and Scaling: Normalization (MinMaxScaler)
# convert categorical to numeric values by one hot encoding
pd.get_dummies(ds_media['date_month'])



# Try Feature Scaling: Normalization (MinMaxScaler) or Standardization
# Feature Scaling when we are dealing with Gradient Descent Based algorithms
# #(Linear and Logistic Regression, Neural Network)
# and Distance-based algorithms (KNN, K-means, SVM) as these are very sensitive to the range of the data points
# This step is not mandatory when dealing with Tree-based algorithms
y = ds_media['revenue']
X=pd.concat([adstock_tv_s,adstock_ooh_s,adstock_print_s,adstock_search_s,adstock_facebook_s,
             ds_media['competitor_sales_b'], pd.get_dummies(ds_media['date_month'])],axis=1)
X.to_pickle('artifacts/model_X.pkl')


X_1 = pd.concat([adstock_tv_s,adstock_ooh_s,adstock_print_s,adstock_search_s,adstock_facebook_s,
             ds_media['competitor_sales_b']],axis=1)


# Normalization
from sklearn.preprocessing import MinMaxScaler
norm = MinMaxScaler()
X_norm_1 = norm.fit_transform(X_1)
X_norm_1 = pd.DataFrame(X_norm_1, columns = X_1.columns)
X_norm =pd.concat([X_norm_1,pd.get_dummies(ds_media['date_month'])],axis=1)
print(X_norm.describe())


# Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std_1 = scaler.fit_transform(X_1)
X_std_1 = pd.DataFrame(X_std_1, columns = X_1.columns)
X_std =pd.concat([X_std_1,pd.get_dummies(ds_media['date_month'])],axis=1)
print(X_std.describe())



# Step 2.2 linear regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Normalization
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_norm, y,random_state=0)
pipeline_lr = Pipeline([('lr_n', LinearRegression())])
pipeline_lr.fit(X_train_n,y_train_n)
pipeline_lr.score(X_test_n, y_test_n)
y_pred_norm = pipeline_lr.predict(X_test_n)

# Score Predictions on Test Set
print(r2_score(y_test_n, y_pred_norm))
print(mean_absolute_error(y_test_n, y_pred_norm))
print(np.sqrt(mean_squared_error(y_test_n, y_pred_norm)))
print(pipeline_lr['lr_n'].coef_)
print(pipeline_lr['lr_n'].intercept_)



# Standardization
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_std, y, random_state=0)
pipeline = Pipeline([('lr_s', LinearRegression())])
pipeline.fit(X_train_s,y_train_s)
pipeline.score(X_test_s, y_test_s)
y_pred_std = pipeline.predict(X_test_s)

# Score Predictions on Test Set
print(r2_score(y_test_s, y_pred_std))
print(mean_absolute_error(y_test_s, y_pred_std))
print(np.sqrt(mean_squared_error(y_test_s, y_pred_std)))
print(pipeline['lr_s'].coef_)
print(pipeline['lr_s'].intercept_)

# no feature scaling
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)
pipeline_1 = Pipeline([('lr', LinearRegression())])
pipeline_1.fit(X_train,y_train)
pipeline_1.score(X_test, y_test)
y_pred = pipeline_1.predict(X_test)

# Score Predictions on Test Set
print(r2_score(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
print(np.sqrt(mean_squared_error(y_test, y_pred)))
print(pipeline_1['lr'].coef_)
print(pipeline_1['lr'].intercept_)

print('No differences between feature scaling and origianl feature; LinearRegression: r2_core = 0.874')



# Step 2.3 Grid Search Adstock Rate(Optimize for Model Fit)
# Generate 100 random numbers between 0 and 1 to find lag weight
rng = np.random.default_rng(123)
size = 100
max_adstock = 1
adstock_grid_df = pd.DataFrame(dict(
    tv=rng.uniform(0, max_adstock, size=size),
    ooh=rng.uniform(0, max_adstock, size=size),
    print=rng.uniform(0, max_adstock, size=size),
    search=rng.uniform(0, max_adstock, size=size),
    facebook=rng.uniform(0, max_adstock, size=size)))

print(adstock_grid_df)



#Step 2.4 Tuning Sklearn Model
# Verbose is a general programming term for produce lots of logging output.
# You can think of it as asking the program to "tell me everything about what you are doing all the time"
# ATTENTION: CANNOT USE PRINT as VARIABLE
def adstock_search(media, adstock_grid, verbose=True):
    # placeholder for model performance
    ds_media = media
    best_model = {'model' : None,
                  'params' : None,
                  'score' : None,
                  'coef' : None}

    # loop each rng (leg weight) rate in the basic adstock function and linear regression model
    for tv, ooh, prnt, search, fb in zip(adstock_grid.tv, adstock_grid.ooh, adstock_grid.print,
                                         adstock_grid.search, adstock_grid.facebook):
        adstock_tv_s = adstock(media.tv_s, tv)
        adstock_ooh_s = adstock(media.ooh_s, ooh)
        adstock_print_s = adstock(media.print_s, prnt)
        adstock_search_s = adstock(media.search_s, search)
        adstock_facebook_s = adstock(media.facebook_s, fb)

        X = pd.concat([adstock_tv_s, adstock_ooh_s, adstock_print_s, adstock_search_s, adstock_facebook_s,
                       media.competitor_sales_b, pd.get_dummies(media.date_month)], axis=1)
        y = media.revenue

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        pipe = Pipeline([('lm', LinearRegression().fit(X_train, y_train))])
        score_1 = pipe.score(X_test, y_test)
        y_pred = pipe.predict(X_test)
        r= r2_score(y_test, y_pred)
        coef = pipe['lm'].coef_


        if best_model['model'] is None or r > best_model['score'] :
            best_model['model'] = pipe
            best_model['params'] = dict(
                tv=tv,
                ooh=ooh,
                print=prnt,
                search=search,
                facebook=fb)
            best_model['score'] = r
            best_model['coef'] = coef

            if verbose:
                print(f'New Best Model:\n{best_model}\n')


    if verbose:
        print('Done!')

    # return best_model

    print(f'\nReport:\nModel Selection:'
          f"\nmodel: {best_model['model']}"
          f"\nr2_score: {best_model['score']}"
          f"\nadstock_rate: \n\t{best_model['params']}"
          f"\ncoef: \n\t{best_model['coef']}")

    return best_model

best_model_sl = adstock_search(ds_media, adstock_grid_df)



# Step 3.0 Budget Rebalance
best_model_coef = best_model_sl['coef']
best_model_coef_names = X.columns

# combine coef_with features and extract only the coef of spending patterns and scaled to percentage of the total spent
# values in each adstock represents the revenue generated for each adstock spent per $1
rebalancing_coef_df = pd.DataFrame(dict(feature = best_model_coef_names, value = best_model_coef))\
                        .set_index('feature')\
                        .filter(regex='_s$', axis=0)\
                        .assign(value = lambda x :x['value']/x['value']\
                        .sum()).reset_index()

print(f'rebalancing_coef: \n{rebalancing_coef_df}')



# Step 3.1 Media Spend Rebalancing
total_spent = df_spent['spent_current'].sum()
df_spent_rebalanced = df_spent.copy() # original dataset will not be affected
df_spent_rebalanced['spend_rebalanced'] = total_spent * rebalancing_coef_df ['value']
print(f'spent_rebalanced: \n{df_spent_rebalanced}')
df_spent_rebalanced.to_pickle('artifacts/spent_rebalanced.pkl')


# Step 3.1.1 Visualize the Media Spending via plotly.express
ds = df_spent_rebalanced.melt(id_vars='media')
fig = px.bar(ds,
				x = 'variable',
				y = 'value',
				color = 'media',
				barmode='group',
				template = 'plotly_dark')

fig.show()



# Step 3.2 Predicted Revenue after Rebalancing
# current revenue vs predict revenue: KPI evaluation
ds_adstock = X.iloc[:,:5]
print(ds_adstock)
ds_ex_adstock = X.iloc[:,5:]

ds_adstock_reb = ds_adstock.copy()

for row, col in enumerate(ds_adstock.columns):
    ds_adstock_reb[col]=ds_adstock_reb.sum(axis=1) * rebalancing_coef_df['value'][row]
print(ds_adstock_reb)
ds_adstock_reb.to_pickle('artifacts/ds_adstock_reb.pkl')


# construct model datasets
X_reb = pd.concat([ds_adstock_reb,ds_ex_adstock],axis=1)
X_reb.to_pickle('artifacts/X_reb.pkl')
X_train_reb, X_test, y_train, y_test = train_test_split(X_reb, y, random_state=0)


# predict_train = predict current revenue
pred_future_revenue_total = best_model_sl['model'][0].predict(X_train_reb).sum()
pred_current_revenue_total = best_model_sl['model'][0].predict(X_train).sum()
improvement = (pred_future_revenue_total - pred_current_revenue_total) / pred_current_revenue_total

print(f'Report:\nBefore Budget Optimization: '
      f'\n\tpred_future_revenue_total:  {pred_future_revenue_total}'
      f'\n\tpred_current_revenue_total: {pred_current_revenue_total}'
      f'\n\timprovement: {improvement}') # 4.95%



# Step 4.0 Budget Optimization --- this is where the money data scientists earn from
# generate 1000 random numbers between 0 and 1 for budget optimization grid search
rng = np.random.default_rng(123)
size = 1000
max=1
budget_grid_df = pd.DataFrame(dict(
    tv=rng.uniform(0, max, size=size),
    ooh=rng.uniform(0, max, size=size),
    print=rng.uniform(0, max, size=size),
    search=rng.uniform(0, max, size=size),
    facebook=rng.uniform(0, max, size=size)))


# Step 4.0.1 Predict Revenue after using the Optimized Rebalancing coef
def optimize_budget(media_spend, budget_grid, verbose=True):
    # placeholder for model performance
    global total_spent_opt, pred_future_revenue_opt_total, pred_current_revenue_total
    best_budget = {'rebalancing_coef_opt':None,
                   'media_budg_opt':None,
                   'improvement_opt':None}
    budget_grid_df =budget_grid
    df_spent = media_spend


    # Scale the random budget mix
    budget_grid_df_scaled = budget_grid_df.copy()
    for n, m in enumerate(budget_grid_df_scaled.index):
        budget_grid_df_scaled.loc[n, :] = budget_grid_df.loc[n, :] / budget_grid_df.loc[n, :].sum()
        # print(budget_grid_df_scaled.loc[n, :])
        budget_scaled = budget_grid_df_scaled.loc[n, :]

        # Create opt.rebalencing coefficients
        rebalancing_coef_opt_df = budget_scaled.to_frame().reset_index().set_axis(['channel', 'value'], axis=1)

        # Rebalance Adstock
        ds_adstock_reb_opt = ds_adstock.copy()
        for row, col in enumerate(ds_adstock_reb_opt.columns):
            ds_adstock_reb_opt[col] = ds_adstock_reb_opt.sum(axis=1) * rebalancing_coef_opt_df['value'][row]


        # construct model datasets
        X_reb_opt = pd.concat([ds_adstock_reb_opt,ds_ex_adstock],axis=1)
        X_train_reb_opt, X_test, y_train, y_test = train_test_split(X_reb_opt, y, random_state=0)


        # predict future revenue after optimization
        pred_future_revenue_opt_total = best_model_sl['model'][0].predict(X_train_reb_opt).sum()
        pred_current_revenue_total = best_model_sl['model'][0].predict(X_train).sum()

        improvement_opt = (pred_future_revenue_opt_total - pred_current_revenue_total) / pred_current_revenue_total


        # Media Spend Optimizied Rebalanced
        total_spent = df_spent['spent_current'].sum()
        df_spent_rebalanced_opt = df_spent.copy()
        df_spent_rebalanced_opt['spend_rebalanced_opt'] = total_spent * rebalancing_coef_opt_df['value']


        if best_budget['improvement_opt'] is None or improvement_opt > best_budget['improvement_opt']:
            best_budget['rebalancing_coef_opt'] = rebalancing_coef_opt_df
            best_budget['media_budg_opt'] = df_spent_rebalanced_opt
            best_budget['improvement_opt'] = improvement_opt

            if verbose:
                print(f'New Best Budget:\n{best_budget}\n')


    if verbose:
        print('Good Job!')

    print(f'Final Report:\nAfter Budget Optimization:'\
          f"\nimprovement_opt: {best_budget['improvement_opt']}" \
          f'\npred_future_revenue_opt_total: {pred_future_revenue_opt_total}' \
          f'\npred_current_revenue_total: {pred_current_revenue_total}'\
          f"\nmedia_budg_opt: \n{best_budget['media_budg_opt']}" \
          f"\ntotal_spent: {best_budget['media_budg_opt']['spend_rebalanced_opt'].sum()}"\
          f"\nrebalancing_coef_opt:\n{best_budget['rebalancing_coef_opt']}")

    return best_budget

best_budget_opt = optimize_budget(df_spent, budget_grid_df)




# # Step 4.0.2 Visualize the Media Spending via plotly.express
ds_opt = best_budget_opt['media_budg_opt'].melt(id_vars='media')
fig = px.bar(ds_opt,
				x = 'variable',
				y = 'value',
				color = 'media',
				barmode='group',
				template = 'plotly_dark')

fig.show()



# Step 5.0 Model persistence
# joblib and pickle
from joblib import dump, load
joblib.dump(best_model_sl, 'artifacts/best_model_joblib')
# print(joblib.load('artifacts/best_model_joblib'))
joblib.dump(best_budget_opt, 'artifacts/best_budget_joblib')
# print(joblib.load('artifacts/best_budget_joblib'))



# "wb" mode opens the file in binary format for writing.
# write the model into binary files
with open('artifacts/best_model_pickle','wb') as f:
    pickle.dump(best_model_sl,f)

# read/load the model in py
with open('artifacts/best_model_pickle','rb') as f:
    best_model = pickle.load(f)
#     print(best_model)

with open('artifacts/best_budget_pickle','wb') as f:
    pickle.dump(best_budget_opt,f)

with open('artifacts/best_budget_pickle','rb') as f:
    best_budget = pickle.load(f)



# Ref: Business Science University ~ LearningLabsPro