# A simple approach for caluating ballpark Customer Lifetime Value (CLV)
## CLV = LTV - Customer Acquisition (AC)
### CLV = Avg of order_value * Num of transaction * Gross Margin * (1 / Churn) - AC

# Assumption:
Gross_Margin = 0.05
AC = 1

import pandas as pd
import numpy as np
from datetime import datetime

# read data source from csv
## convert orderdate from object to datetime
### %H Hour (24-hour clock) as a decimal number [00,23]
### %I Hour (12-hour clock) as a decimal number [01,12]
### %p AM or PM
### encoding = 'ISO-8859-1' equals to 'latin-1'
### default header = 0 means the first row of the CSV file will be treated as column names
dates = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

df_order = pd.read_csv('data/orders.txt', sep='\t', encoding = "latin-1", date_parser= dates,parse_dates=['orderdate'])
df_customer = pd.read_csv('data/customer.txt', sep='\t', encoding = "latin-1")

# output csv file
## repr（）：printable representation of the given object
### print the whole content of the dataset: set_option; display.max; display.expend_frame_repr
pd.set_option('display.max_column', None, 'display.expand_frame_repr', False)

df_order
df_order.info()
df_order.columns.tolist()
df_customer
df_customer.dtypes
df_customer.info()
df_customer.columns.tolist()

# extract useful imputs from both ds
df_order = df_order[['orderid', 'customerid', 'orderdate', 'totalprice']]
df_customer = df_customer[['customerid','householdid']]

# group by customerid and perform aggregations
order_lenth= {'orderdate':lambda x: (x.max() -x.min()).days}
total_sales = {'totalprice':lambda x: x.sum()}
num_of_orders ={'orderid': lambda x:len(x)}

df_t= df_order.groupby('customerid').agg(total_sales)
df_n= df_order.groupby('customerid').agg(num_of_orders)
# filter dates >0
df_r= df_order.groupby('customerid').agg(order_lenth)
df_r=df_r[df_r['orderdate']>0]

# concatenate three dataframes either merge or concat
## customer_info = pd.merge(pd.merge(df_t,df_n, on='customerid'),df_r, on='customerid')
dfs = [df_r,df_t,df_n]
customer_info = pd.concat(dfs,axis=1, join= 'inner', levels=0).reset_index()
customer_info.columns = ['customerid', 'order_lenth','total_sales','num_of_orders']
print(customer_info)

# CLV = Avg of order_value * Num of transaction * Gross Margin * (1 / Churn) - AC
## iloc/loc accessing multiindex df
avg_order_value = customer_info.iloc[0,2] / customer_info.iloc[0,3]

## frequency = total number of orders / order_lenth
frequency = customer_info.iloc[0,3].sum()/ customer_info.iloc[0,1].sum()

## retention is the percentage of customers who ordered more than once; churn = 1- retention
### shape[0] in pandas means the records - row in shape
retention = df_n[df_n['orderid']>1].shape[0] / df_n.shape[0]

clv = avg_order_value * frequency / (1-retention) * Gross_Margin - AC
df = pd.DataFrame({'simple CLV': clv,
                   'avg_order_value':round(avg_order_value,3),
                   'frequency': round(frequency,3),
                   'retention':retention,
                   'gross_margin':Gross_Margin,
                   'customer_acquisition':AC}, index = ['result'])
print(df)
