# RFM Analysis via auto clustering
# Part One: get RFM clustered via Kmeans



import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


# default='warn', dealing with SettingWithCopyWarning
pd.options.mode.chained_assignment = None


# read data source from csv
## convert orderdate from object to datetime
### %H Hour (24-hour clock) as a decimal number [00,23]
### %I Hour (12-hour clock) as a decimal number [01,12]
### %p AM or PM
### encoding = 'ISO-8859-1' equals to 'latin-1'
### default header = 0 means the first row of the CSV file will be treated as column names
dates = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

df_order = pd.read_csv('data/orders.txt', sep='\t', encoding = "latin-1", date_parser= dates,parse_dates=['orderdate'],
                       infer_datetime_format=True)
df_customer = pd.read_csv('data/customer.txt', sep='\t', encoding = "latin-1")



# output csv file
## repr（）：printable representation of the given object
### print the whole content of the dataset: set_option; display.max; display.expend_frame_repr
pd.set_option('display.max_column', None, 'display.expand_frame_repr', False)


df_order =df_order.drop(['campaignid', 'city', 'state', 'zipcode','paymenttype','numorderlines', 'numunits'], axis=1)
print(df_order.info())
df_customer = df_customer.drop(['gender', 'firstname'], axis = 1)
print(df_customer.info())



# householdid has null value if how = 'left'
df = pd.merge(df_order,df_customer,left_on='customerid', right_on='customerid')

# group by householdid
# recency: the max order date of the whole transaction minus the max order date of each group
df_rfm = df.groupby('householdid').agg({'orderid': 'count',
                               'totalprice' : 'sum',
                               'orderdate' : lambda x : (df['orderdate'].max() - x.max()).days}).reset_index()
df_rfm.columns = [['householdid','frequency','monetary','recency']]
df_f = df_rfm[['householdid','frequency']]
df_m = df_rfm[['householdid','monetary']]
df_r = df_rfm[['householdid','recency']]


# auto-clustering for r,f,m individually
from sklearn.cluster import KMeans
SSE= []
for n in range(1,21):
    kmeans = KMeans(n_clusters=n, random_state=42).fit(df_r)
    # The lowest Error Sum of Squares(SSE) value
    SSE.append(kmeans.inertia_)
    print(SSE)



# SSE curve starts to bend known as the elbow point.
# The x-value of this point is thought to be a reasonable trade-off between error and number of clusters.
plt.style.use("fivethirtyeight")
plt.plot(range(1, 21), SSE)
plt.xticks(range(1, 21))
plt.xlabel("Number of Clusters for recency")
plt.ylabel("SSE")
plt.show()



# use a Python package, kneed, to identify the elbow point programmatically
from kneed import KneeLocator
kl = KneeLocator(range(1,21),SSE, curve = 'convex', direction='decreasing')
print(kl.elbow)



# determine n_clusters for recency by elbow point +1 = 3+1 = 4
kmeans = KMeans(n_clusters=4, random_state=42).fit(df_r['recency'].to_numpy())
pred_r=kmeans.predict(df_r['recency'].to_numpy())
df_r['cluster_r'] =pred_r



# clustering frequency
SSE= []
for n in range(1,21):
    kmeans = KMeans(n_clusters=n, random_state=42).fit(df_f.to_numpy())
    # The lowest Error Sum of Squares(SSE) value
    SSE.append(kmeans.inertia_)
    print(SSE)


plt.style.use("fivethirtyeight")
plt.plot(range(1, 21), SSE)
plt.xticks(range(1, 21))
plt.xlabel("Number of Clusters for frequency")
plt.ylabel("SSE")
plt.show()


from kneed import KneeLocator
kl = KneeLocator(range(1,21),SSE, curve = 'convex', direction='decreasing')
print(kl.elbow)


# determine n_clusters for frequency by elbow point +1 = 3+1 = 4
kmeans = KMeans(n_clusters=4, random_state=42).fit(df_f['frequency'].to_numpy())
pred_f=kmeans.predict(df_f['frequency'].to_numpy())
df_f['cluster_f'] =pred_f



# clustering monetary
SSE= []
for n in range(1,21):
    kmeans = KMeans(n_clusters=n, random_state=42).fit(df_m.to_numpy())
    # The lowest Error Sum of Squares(SSE) value
    SSE.append(kmeans.inertia_)
    print(SSE)


plt.style.use("fivethirtyeight")
plt.plot(range(1, 21), SSE)
plt.xticks(range(1, 21))
plt.xlabel("Number of Clusters for monetary")
plt.ylabel("SSE")
plt.show()


from kneed import KneeLocator
kl = KneeLocator(range(1,21),SSE, curve = 'convex', direction='decreasing')
print(kl.elbow)



# determine n_clusters for monetary by elbow point +1 = 3+1 = 4
kmeans = KMeans(n_clusters=4, random_state=42).fit(df_m['monetary'].to_numpy())
pred_m=kmeans.predict(df_m['monetary'].to_numpy())
df_m['cluster_m'] =pred_m



# save all fies to .csv 
df_r.to_csv('data/df_r')
df_f.to_csv('data/df_f')
df_m.to_csv('data/df_m')






# RFM Analysis via auto clustering
# Part Two: indexing RFM clusters and implement business objectives


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df_r = pd.read_csv('data/df_r', sep=',').drop(['Unnamed: 0'], axis = 1)
df_f = pd.read_csv('data/df_f', sep=',').drop(['Unnamed: 0'], axis = 1)
df_m = pd.read_csv('data/df_m', sep=',').drop(['Unnamed: 0'], axis = 1)



''' 
   cluster_r      recency
0          3   285.400161
1          1   832.895055
2          0  1512.677244
3          2  2161.399416

   cluster_f   frequency
0          2    1.000000
1          0    2.474662
2          3  169.000000
3          1  746.000000

   cluster_m     monetary
0          1    44.254745
1          0   468.018544
2          2  1711.942747
3          3  5334.465287
it clearly states that the value of cluster_m is nominal instead of ordinal  '''



# clustering labeled R,F,M are nominal value for presenting
# convert nominal into ordinal value
# cluster_field_name = [cluster_r,cluster_f,cluster_m]; target_field_name =[r,f,m]
# compare the mean of each cluster and sort in ascending
# reindex the recency according to the mean value
def order_cluster(cluster_field_name, target_field_name, df):
    ''' sort each cluster by its mean'''
    df_sorted = df.groupby(cluster_field_name).agg({target_field_name : "mean"}).\
        sort_values(by=target_field_name,ascending=True).reset_index()

    '''pass index according to the mean'''
    df_sorted['index']=df_sorted.index

    '''replace the cluster label with the sorted index'''
    df_new = pd.merge(df, df_sorted[[cluster_field_name, 'index']], on=cluster_field_name).drop([cluster_field_name],
                      axis =1).rename(columns={'index':cluster_field_name})
    return df_new

df_r_index=order_cluster('cluster_r','recency',df_r)
df_f_index=order_cluster('cluster_f','frequency',df_f)
df_m_index=order_cluster('cluster_m','monetary',df_m)



# drop the unwanted columns
df_r_index=df_r_index[['householdid','cluster_r']]
df_f_index=df_f_index[['householdid','cluster_f']]
df_m_index=df_m_index[['householdid','cluster_m']]



# three-way joining multiples df on columns
from functools import reduce
data_frames = [df_r_index, df_f_index, df_m_index]
df_rfm_index=reduce(lambda left,right: pd.merge(left,right,on=['householdid']), data_frames)
print(df_rfm_index)

# save file to .csv
df_rfm_index.to_csv('data/rfm_clusted')


'''
Define business objectives for clusters:
Recency	                Frequency	                Monetary
R-Tier-3 (most recent)	F-Tier-3 (most frequent)	M-Tier-3 (highest spend)
R-Tier-2	            F-Tier-2	                M-Tier-2
R-Tier-1	            F-Tier-1	                M-Tier-1
R-Tier-0 (least recent)	F-Tier-0 (least frequency)	M-Tier-0 (lowest spend)     
'''

print('churned customers with no records: 0-0-0; 26130 in 156258')
df_rfm_churned=df_rfm_index[(df_rfm_index['cluster_r'] == 0) & (df_rfm_index['cluster_f'] ==0) & (df_rfm_index['cluster_m']==0)]
print(df_rfm_churned.count())




print('best customers: 3-3-3; 3-3-2; (0)')
df_rfm_best=df_rfm_index[(df_rfm_index['cluster_r'] == 3) & (df_rfm_index['cluster_f'] ==3) & (df_rfm_index['cluster_m']>=2)]
print(df_rfm_best.count())




print('high-spending new customers: 3-0-3:(0) and 3-0-2: (17)')
df_rfm_best_new=df_rfm_index[(df_rfm_index['cluster_r'] == 3) & (df_rfm_index['cluster_f'] ==0) & (df_rfm_index['cluster_m'] >= 2)]
print(df_rfm_best_new.count())




print('low-spending active loyal customers: 3-3-0; 3-3-1; 2-2-0; 2-2-1; (0)')
df_rfm_loyal_low_1=df_rfm_index[(df_rfm_index['cluster_r'] == 3) & (df_rfm_index['cluster_f'] == 3) & (df_rfm_index['cluster_m'] ==0)]
df_rfm_loyal_low_2=df_rfm_index[(df_rfm_index['cluster_r'] == 3) & (df_rfm_index['cluster_f'] == 3) & (df_rfm_index['cluster_m'] ==1)]
df_rfm_loyal_low_3=df_rfm_index[(df_rfm_index['cluster_r'] == 2) & (df_rfm_index['cluster_f'] == 2) & (df_rfm_index['cluster_m'] ==0)]
df_rfm_loyal_low_4=df_rfm_index[(df_rfm_index['cluster_r'] == 2) & (df_rfm_index['cluster_f'] == 2) & (df_rfm_index['cluster_m'] ==1)]
print(df_rfm_loyal_low_1.count())
print(df_rfm_loyal_low_2.count())
print(df_rfm_loyal_low_3.count())
print(df_rfm_loyal_low_4.count())



print('churned best customers: 0-3-3;(1), 0-3-2; 0-2-3; 0-2-2')
df_rfm_churned_best_1 = df_rfm_index[(df_rfm_index['cluster_r'] == 0) & (df_rfm_index['cluster_f'] ==3) & (df_rfm_index['cluster_m'] ==3)]
df_rfm_churned_best_2 = df_rfm_index[(df_rfm_index['cluster_r'] == 0) & (df_rfm_index['cluster_f'] ==3) & (df_rfm_index['cluster_m'] ==2)]
df_rfm_churned_best_3 = df_rfm_index[(df_rfm_index['cluster_r'] == 0) & (df_rfm_index['cluster_f'] ==2) & (df_rfm_index['cluster_m'] ==3)]
df_rfm_churned_best_4 = df_rfm_index[(df_rfm_index['cluster_r'] == 0) & (df_rfm_index['cluster_f'] ==2) & (df_rfm_index['cluster_m'] ==2)]
print(df_rfm_churned_best_1)
print(df_rfm_churned_best_2.count())
print(df_rfm_churned_best_3.count())
print(df_rfm_churned_best_4.count())
