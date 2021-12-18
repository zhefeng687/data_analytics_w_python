# Survival analysis is a set of statistical approaches used to determine the time it takes for an event of interest to occur.
# We use survival analysis to study the time until some event of interest occurs.
# Time is usually measured in years, months, weeks, days, and other time measuring units.
# The event of interest could be anything of interest.
# It could be an actual death, a birth, a retirement, along with others.


#1) Kaplan-Meier plots to visualize survival curves.
#2) Nelson-Aalen plots to visualize the cumulative hazard.
#3) Log-Rank test to compare the survival curves of two or more groups
#4) Cox-proportional hazards regression finds out the effect of different variables on survival.


# Part 1 POC: KM Estimator for customer survival; NA Estimator for customer churn
# Part 2 Segment customer by rate plans and apply to survival and churn analysis
# Part 3 Log Rank to test the significant difference among groups
# Part 4 The effect of each factor for the survival analysis


# Part 1 POC
# event 'death' means customer churned
# First discuss the overall customer retention without segmentation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


pd.set_option('display.max_column', None, 'display.expand_frame_repr', False)
df = pd.read_csv('data\subs.txt', sep= '\t', encoding = 'Latin-1', parse_dates = ['start_date', 'stop_date'])
print(df.head())
print(df.columns)
print(df.info())

print(df['tenure'].describe())
print(df['tenure'].hist())

print(df['rate_plan'].unique())
df['rate_plan'].hist()
plt.show()
print(df['market'].unique())
df['market'].hist()
plt.show()
print(df['channel'].unique())
df['channel'].hist()
plt.show()
print(df['stop_type'].unique())
df['stop_type'].hist()
plt.show()


# Data preprocessing for missing value
# check null value: isnull() return F or T; F = 0; T =1
print(df['start_date'].isnull().values.sum())


# deal with left-censoring data
# select tenure >0 and start_date is after 2004-01-01
df = df.query("tenure >= 0 and start_date > '2004-01-01'")
print(df.head())


# construct event_observed : churned column
# stop type is null, which indicates the customer is still alive
df['churned'] = df['stop_type'].apply(lambda x: 0 if pd.isnull(x) else 1)


# observed data stores the value of dead persons in a specific timeline
# censored data stores the value of alive persons or persons
# that we are not going to investigate
print(df[['churned','censored']])


# KaplanMeier tells the probability of the event of interest
# (customer churned in our case) not occuring by that time
from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()
kmf.fit(durations = df['tenure'],event_observed = df['churned'])
print(kmf.survival_function_)


# event table
# removed = observed + censored
# censored = customers did not churn
# observed = customers did churn
print(kmf.event_table)


kmf.plot_survival_function()
plt.title('KM Estimate of Customer Subs')
plt.xlabel('Number of Days')
plt.ylabel('Probability of Survival')
plt.show()


# The median number of survival days
print('The median survival time:', kmf.median_survival_time_)


# estimate the survival probability with confidence interval: 0.95
print(kmf.confidence_interval_survival_function_)
confidence_surv_func = kmf.confidence_interval_survival_function_
plt.plot(confidence_surv_func['KM_estimate_upper_0.95'], label = 'Upper')
plt.plot(confidence_surv_func['KM_estimate_lower_0.95'], label ='Lower')
plt.title('Survival Function with Confidence Interval')
plt.xlabel('number of days')
plt.ylabel('Probability of Survival')
plt.legend()
plt.show()


# probability of a customer churned as the cumulative density
print(kmf.cumulative_density_)
kmf.plot_cumulative_density()
plt.title('Cumulative Density Plot')
plt.xlabel('Number of days')
plt.ylabel("Probability of a customer's churn")
plt.show()


# cumulative density with confidence interval
print(kmf.confidence_interval_cumulative_density_)


# the amount of time remaining from the median survival time (568 days)
print(kmf.conditional_time_to_event_)
median_time_to_event = kmf.conditional_time_to_event_
plt.plot(median_time_to_event, label = 'Median time left')
plt.title("Medain time to event")
plt.xlabel("Total days")
plt.ylabel("Conditional median time to event")
plt.legend()
plt.show()


# Estimate hazard rates using Nelson-Aalen
# The hazard function h(t) gives us the probability
# that a subject under observation at time t has an event of interest (death) at that time.
from lifelines import NelsonAalenFitter
naf = NelsonAalenFitter()
naf.fit(df['tenure'], event_observed=df['churned'])


# Print the cumulative hazard:
print(naf.cumulative_hazard_)


# Plot the cumulative hazard grpah:
naf.plot_cumulative_hazard()
plt.title("Cumulative Probability for customer churned")
plt.xlabel("Number of days")
plt.ylabel("Cumulative Probability of customer churned")
plt.show()


# Cumulative hazard with confidence interval:
naf.confidence_interval_


# Plot cumulative hazard with confidence interval:
confidence_interval = naf.confidence_interval_
plt.plot(confidence_interval["NA_estimate_lower_0.95"],label="Lower")
plt.plot(confidence_interval["NA_estimate_upper_0.95"],label="Upper")
plt.title("Cumulative hazard With Confidence Interval")
plt.xlabel("Number of days")
plt.ylabel("Cumulative hazard")
plt.legend()


# Plot the cumulative_hazard and cumulative density:
kmf.plot_cumulative_density(label="Cumulative Density")
naf.plot_cumulative_hazard(label="Cumulative Hazard")
x1=plt.xlabel("Number of Days")
x1=plt.title("Cumulative Density vs Cumulative hazard")
x1=plt.show()




# Part 2 Segment customer by rate plans and apply to survival and churn analysis
# Event 'death' means customer churned
# First discuss the overall customer retention without segmentation
# customers group by rate plans
Bottom = df.query(" rate_plan == 'Bottom'")
Middle = df.query(" rate_plan == 'Middle'")
Top = df.query(" rate_plan == 'Top'")
print(Bottom.head())
print(Middle.head())
print(Top.head())
print(df['monthly_fee'].unique())
print(df['monthly_fee'].describe())


# fit data into KM model
from lifelines import KaplanMeierFitter
kmf_Bottom = KaplanMeierFitter()
kmf_Bottom.fit(durations = Bottom['tenure'], event_observed = Bottom['churned'], label = 'Bottom')

kmf_Middle = KaplanMeierFitter()
kmf_Middle.fit(durations = Middle['tenure'], event_observed = Middle['churned'], label = 'Middle')

kmf_Top = KaplanMeierFitter()
kmf_Top.fit(durations = Top['tenure'], event_observed = Top['churned'], label = 'Top')


# event table
kmf_Bottom.event_table
kmf_Middle.event_table
kmf_Top.event_table


# Survival probability for each group:
print(kmf_Bottom.survival_function_)
print(kmf_Middle.survival_function_)
print(kmf_Top.survival_function_)


# Plot the graph for survival probabilities:
kmf_Bottom.plot()
kmf_Middle.plot()
kmf_Top.plot()
plt.xlabel('Days Passed')
plt.ylabel('Survival Probability')
plt.title('KMF')
plt.show()


# The median number of survival days
print('The median survival time for bottom plan:', kmf_Bottom.median_survival_time_)
print('The median survival time for middle plan:', kmf_Middle.median_survival_time_)
print('The median survival time for top plan:', kmf_Top.median_survival_time_)


#Hazard Function:
from lifelines import NelsonAalenFitter
naf_Bottom = NelsonAalenFitter()
naf_Middle = NelsonAalenFitter()
naf_Top = NelsonAalenFitter()

naf_Bottom.fit(Bottom['tenure'], event_observed = Bottom['churned'],label = 'Bottom')
naf_Middle.fit(Middle['tenure'], event_observed = Middle['churned'], label = 'Middle')
naf_Top.fit(Top['tenure'], event_observed = Top['churned'], label = 'Top')


#Cumulative hazard probability for three groups:
print(naf_Bottom.cumulative_hazard_)
print(naf_Middle.cumulative_hazard_)
print(naf_Top.cumulative_hazard_)


#Plot the graph for cumulative hazard:
naf_Bottom.plot_cumulative_hazard(label = 'Bottom')
naf_Middle.plot_cumulative_hazard(label = 'Middle')
naf_Top.plot_cumulative_hazard(label = 'Top')
plt.title('Cumulative Hazard Plot')
plt.xlabel('Days Passed')
plt.ylabel('Cumulative Hazard')
plt.show()




# Part 3 Log-Rank test to compare the survival curves of two or more groups
# Goal: Our goal is to see if there is any significant difference between the groups being compared.

# Null Hypothesis:
# The null hypothesis states that there is no significant difference between the groups being studied.
# If there is a significant difference between those groups, then we have to reject our null hypothesis.
# A p-value between 0 and 1 denotes the statistical significance.
# The smaller the p-value, the more significant the statistical difference between groups being studied is.
# Less than (5% = 0.05) P-value means a significant difference between the groups we compared.


# Define variables for log-rank test
Duration_Bottom = Bottom['tenure']
Event_Bottom = Bottom['churned']

Duration_Middle = Middle['tenure']
Event_Middle = Middle['churned']

Duration_Top = Top['tenure']
Event_Top = Top['churned']


# Perform the log-rank test
from lifelines.statistics import logrank_test
results_bm = logrank_test(Duration_Bottom,Duration_Middle,event_observed_A=Event_Bottom,event_observed_B=Event_Middle)
results_bt = logrank_test(Duration_Bottom,Duration_Top,event_observed_A=Event_Bottom,event_observed_B=Event_Top)
results_tm = logrank_test(Duration_Top,Duration_Middle,event_observed_A=Event_Top,event_observed_B=Event_Middle)
results_bm.print_summary()
results_bt.print_summary()
results_tm.print_summary()


# p value
print('P-value_bm :', results_bm.p_value)
print('P-value_bt:', results_bt.p_value)
print('P-value_tm:', results_tm.p_value)


# all p-value are less than 0.05
# which denotes that we have to reject the null hypothesis,
# significantly different for all three plans in survival probability


# In short, rate_plan of a customer makes a significant difference in survival probability




# Part 4 The effect of each factor for the survival analysis
# Cox-proportional hazards regression finds out
# How different parameters affects the survival time of a subject in terms of coefficients
# Works for both quantitative predictors non-categorical and categorical variables.

# Convert categorical data into numeric data for CoxPH Fitter
from lifelines import CoxPHFitter
# check null data
df=df.drop(['customer_id','start_date','stop_date', 'stop_type','censored'],axis = 1)
print(df.info())
print(df.isnull().values.sum())


# create a copy of the data with only the object columns.
obj_df = df.select_dtypes(include=['object']).copy()
print(obj_df.head())


# check for null values in the dataset
print(obj_df[obj_df.isnull().any(axis=1)])


# Label Encoding: converting each value in a column to a number.
# But it is ordinal, numeric values can be “misinterpreted” by the algorithms
# Using One Hot Encoding (get_dummies) instead
# Convert each category value into a new column,
# Assigns a 1 or 0 (True/False) value to the column
print(obj_df['rate_plan'].value_counts())
print(obj_df['market'].value_counts())
print(obj_df['channel'].value_counts())


# One Hot Encoding (get_dummies)
obj_df_n = pd.get_dummies(obj_df, columns=['rate_plan','market','channel'])
print(obj_df_n.head())


# filter the numeric column
df_n = df.select_dtypes(include=['int64']).copy()
print(df_n.head())


# join the encoded data back to the original dataframe
data = pd.concat([obj_df_n,df_n], axis=1)
print(data.head())
print(data.info())


# fit the model
# KaplanMeierFitter
kmf = KaplanMeierFitter()
kmf.fit(durations = data['tenure'], event_observed = data['churned'])
print(kmf.event_table)


# Survival probability
print(kmf.survival_function_)


# Plot the graph for survival probabilities:
kmf.plot()
plt.xlabel('Days Passed')
plt.ylabel('Survival Probability')
plt.title('KMF')
plt.show()


# The median number of survival days
print('The median survival time for bottom plan:', kmf.median_survival_time_)


# Get the summary using CoxPHFitter:
# The standard Cox Regression fails if features are highly correlated: Multicollinearity
# Invertibility is solved in linear regression using regularisation
# https://lifelines.readthedocs.io/en/latest/Examples.html
# HR(Hazard Ratio) = exp(bi)
# HR=1: No effect; HR<1: Reduction in the hazard; HR>1: Increase in hazard
cph = CoxPHFitter(penalizer=0.1)
cph.fit(data,'tenure',event_col='churned')
cph.print_summary()


# Plot the result on graph:
cph.plot()
plt.show()


# Other than rate_plan_Top, the P values for the rests are all smaller than 0.05.
# Hazard Ratio are 1.21(Gotham), 1.16(Chain), 1.09(Dealer), 1.13(Mail), respectively indicating a strong probability of increased risk of churn, (21%, 16%, 9%, 13%).
# Hazard Ratio are 0.68(Smallville), 0.69(Store) indicating a strong probability of reduced risk of churn, (32%, 31%).


# Plot the survival function
d_data = data.iloc[12:17, :]
cph.predict_survival_function(d_data).plot()
plt.show()


# find out the median time to event (churn) for timeline:
# notice that as the number of days passed within a year [365 days],
# the median survival time is decreasing
CTE = kmf.conditional_time_to_event_
plt.plot(CTE)
plt.show()
