# Cohort analysis, a behavioral analytical technique used to track specific metrics based on the groups of users
## created by the time of their first actions within the program or website, (either first appereacnec on the website, registration or purchase)


# Business Value
## Cohort analysis is used in transaction-based type of business
## Cohort analysis can be helpful when it comes to understand the 'health and stickness' of your business - the loyalty of your customers
## Stickness is critical since it's far cheaper and easier to keep a current customer than to acuqire a new one
## Also, your product evolves over time. New features are added and removed, the design changes, etc.
## Observing individual groups over time is a starting point to understanding how these changes affect user behavior
## Household level is always welcomed
## Cohort analysis requires the completeness of data



# read data source from csv
## convert orderdate from object to datetime
import pandas as pd
import numpy as np

df_orders = pd.read_csv('data/orders.txt', sep='\t', encoding= 'latin-1',parse_dates=['orderdate'],
						infer_datetime_format=True)
df_orders = df_orders[['orderid', 'customerid','orderdate','totalprice']]
print(df_orders)

df_customer = pd.read_csv('data/customer.txt', sep='\t', encoding= 'latin-1')
df_customer = df_customer[['customerid','householdid']]
print(df_customer)


# be careful using concat, it can have duplicate index since you dont specify the columns to 'join'
# using merge instead, two ways for inner join: how, on ; left_on, right_on
## noted that df_orders has more records than df_customers: df_orders needs to put first than df_customers
## left join and inner join with on ='' both works
df = pd.merge(df_orders, df_customer, how = 'inner', on='customerid')
print(df)


# Data preprocessing for missing value
## check null value: isnull() return F or T; F = 0; T =1
df = df.drop(['customerid'], axis = 1)
print(df.isnull().sum())
print(df.isnull().values.sum())


# EDA
## inspect how many orders were placed by each household
## inspect the percentage of household ordered more than once
print(df['totalprice'].describe())
n_orders = df.groupby(['householdid'])['orderid'].nunique()
print(n_orders)
mult_orders_perh = round((np.sum(n_orders > 1) / df['householdid'].nunique())*100,2)
print(f'{mult_orders_perh}% of households ordered more than once.')


# Data wrangling
# here using month buckets to deal with cohort analysis
## extract month and year separtely; since we are only dealing with month, replace day by1
### df['date_column_trunc'] = df[date_column'].apply(lambda s: datetime.date(s.year, s.month, 1)
import datetime as dt
df['order_month'] = df['orderdate'].apply(lambda x : dt.datetime(x.year, x.month,1))
print(df)


# look for the monthly first order date for households, which indicates when we acquired them
df_gb_household=df.groupby(['householdid'])['order_month'].min().reset_index()
df_gb_household.columns = ['householdid','cohort']
df = pd.merge(df,df_gb_household, how = 'inner', on='householdid')
# df_cohort_1=pd.merge(df,df_gb_household, left_on='householdid', right_on='householdid')
print(df)


# calculate period_month : indicates the number of periods in months between the cohort month and the month of the purchase.
## add 1 when two are in the same month
df['period_month'] = df['order_month'].apply(lambda x: x.year * 12 + x.month) - \
					 df['cohort'].apply(lambda x: x.year * 12 + x.month)
pd.set_option('display.max_column', None, 'display.expand_frame_repr', False)
print(df)


# aggregate the data per cohort and period_month
# count the number of unique households in each group
df_cohort_analysis= df.groupby(['cohort','period_month']).agg(num_households=('householdid','nunique')).reset_index()
print(df_cohort_analysis)
df_cohort_analysis['cohort'] = df_cohort_analysis['cohort'].dt.strftime('%Y/%m')


# pivot the table in a way that each row contains information about a given cohort
# each column contains values for a certain period in month
cohort_analysis = df_cohort_analysis.pivot_table(index = 'cohort', columns='period_month', values ='num_households')
print(cohort_analysis)


# retention matrix calculation
## we need to divide the values in each by its first value,
## which is actually the cohort size — all households who made their first purchase in the given month
cohort_size = cohort_analysis.iloc[:,0]
retention_matrix = cohort_analysis.divide(cohort_size, axis = 0).round(4)*100
print(retention_matrix)


# plot retention matrix plus extra information regarding the cohort size
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

with sns.axes_style("white"):
	fig, ax = plt.subplots(1, 2,sharey=True, figsize=(12,10),gridspec_kw={'width_ratios': [1, 25]})

	# retention matrix
	sns.heatmap(retention_matrix.iloc[0:12,0:25],
				mask=retention_matrix.iloc[0:12,0:25].isnull(),
				cmap='YlGnBu',
				annot=True,
				vmin=0.0,
				vmax=20,
				fmt='g',
				ax=ax[1])
	ax[1].set_xlabel('# of Periods', fontsize=12)
	ax[1].set_ylabel('')
	ax[1].set_title('Monthly Cohorts: User Retention (month 0 - 24)', fontsize = 14)


	# cohort size
	cohort_size = cohort_analysis.iloc[0:12,0]
	cohort_size_df = pd.DataFrame(cohort_size).rename(columns={0: 'cohort_size'})
	white_cmap = mcolors.ListedColormap(['white'])
	sns.heatmap(cohort_size_df,
				annot=True,
				cbar=False,
				fmt='g',
				cmap=white_cmap,
				ax=ax[0])
	ax[0].tick_params(axis='y', rotation=0)
	ax[0].set_ylabel('Cohort Group', fontsize=12)
	fig.tight_layout()

plt.show()