
from tslearn.clustering import TimeSeriesKMeans, KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax, TimeSeriesResampler
from tslearn.utils import to_time_series_dataset
from sqlalchemy import create_engine
from IPython import get_ipython
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float

import pandas as pd
import pycaret
import sqlalchemy
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')

startDate = "2020-01-01 00:00:00"
endDate = "2020-02-01 00:00:00"
id_unit_usaha = '024'

login = ""
password = ""
#engine = sqlalchemy.create_engine('mysql+pymysql://energy:energy2x5=10@localhost:3306/pgn')
engine = sqlalchemy.create_engine(
    'mssql+pyodbc://sa:ams123@10.147.18.38/SIPG?driver=SQL+Server')

sql = " SELECT a.IDREFPELANGGAN, a.ID_UNIT_USAHA, 1 AS FSTREAMID, DATEPART(dw, a.FDATETIME) as FDAYOFWEEK, a.FHOUR, AVG(a.FDVC) as AVG_FDVC\
            FROM(SELECT IDREFPELANGGAN, ID_UNIT_USAHA, FDATETIME, FHOUR, SUM(FDVC) as FDVC\
                FROM amr_bridge\
                WHERE FDATETIME >= '" + startDate + "'\
                and FDATETIME < '" + endDate + "'\
                GROUP BY IDREFPELANGGAN, ID_UNIT_USAHA, FDATETIME, FHOUR) a\
            GROUP BY a.IDREFPELANGGAN, a.ID_UNIT_USAHA, DATEPART(dw, a.FDATETIME), a.FHOUR\
            ORDER BY a.IDREFPELANGGAN, a.ID_UNIT_USAHA, DATEPART(dw, a.FDATETIME), a.FHOUR"

#FDATETIME, FDATE, FDATETIME, FDATE,
# ORDER BY IDREFPELANGGAN, FDATETIME "

df = pd.read_sql_query(sql, engine)


'''
def select_data(start_date, end_date, id_unit):
    query = "ID_UNIT_USAHA == '{}' and FDATETIME >='{}' and FDATETIME <= '{}' ".format(id_unit_usaha, start_date, end_date)
    #query = "ID_UNIT_USAHA.str.contains('{}') and FDATETIME >='{}' and FDATETIME <= '{}' ".format(id_unit_usaha, start_date, end_date)
    columns = ['FDATETIME', 'FDATE', 'FDAYOFWEEK', 'FMONTH', 'FYEAR', 'FHOUR','IDREFPELANGGAN', 'FDVC']
    
    #df = df.set_index('FDATETIME')
    df_selected = df.query(query, engine='python')[columns]
    return df_selected
'''


def select_data(id_unit):
    query = "ID_UNIT_USAHA == '{}'".format(id_unit_usaha)
    columns = ['FDAYOFWEEK', 'FHOUR', 'IDREFPELANGGAN', 'AVG_FDVC']

    #df = df.set_index('FDATETIME')
    df_selected = df.query(query, engine='python')[columns]
    return df_selected


def pivot_data(df):
    #df_pivoted = df.pivot(index='FDATETIME', columns='IDREFPELANGGAN', values='FDVC')
    df_pivoted = df.pivot(
        index=['FDAYOFWEEK', 'FHOUR'], columns='IDREFPELANGGAN', values='AVG_FDVC')
    return df_pivoted


def remove_zerocolumns(df):
    # Get all columns which have all zero values
    cols = df.columns[df.mean() == 0]
    # Drop columns which has all zero values
    df = df.drop(cols, axis=1)
    return df


#start_date = '2020-11-02 00:00:00'
#end_date = '2020-11-8 23:00:00'

df_week1 = select_data(id_unit_usaha)
df_week1.fillna(0.0, inplace=True)


# Pivot table
df_pivoted1 = pivot_data(df_week1)
df_pivoted1.fillna(0.0, inplace=True)
df_pivoted1


# Remove zero columns
df_pivoted1 = remove_zerocolumns(df_pivoted1)
cols = list(df_pivoted1.columns)
df_pivoted1.head()


scaler = StandardScaler()
df_pivoted1_norm = scaler.fit_transform(df_pivoted1)


df_norm = pd.DataFrame(df_pivoted1_norm, columns=cols, index=df_pivoted1.index)
df_norm


# Convert normalized data frame to list of series. Each column is a series.
norm_series = []
for i, y in enumerate(cols):
    length = len(df_norm[y])
    cst = df_norm[y].values.reshape(length, 1)
    norm_series.append(cst)

# convert list to array
norm_series_array = np.array(norm_series)


# Function to plot cluster
def plot_clusters(ds, y_pred, n_clusters, ks, filename):
    plt.figure(figsize=(12, 40))
    for yi in range(n_clusters):
        plt.subplot(n_clusters, 1, 1 + yi)
        for xx in ds[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(ks.cluster_centers_[yi].ravel(), "r-")
        plt.xlim(0, sz)
        plt.ylim(-7, 7)
        plt.title("Cluster %d" % (yi))

    plt.tight_layout()
    plt.savefig(filename, format='jpg', dpi=300, quality=95)
    plt.show()


# Create data frame for customer and its cluster
def create_cluster_info(y_pred, cols):

    df_cluster = pd.DataFrame(
        y_pred.copy(), index=cols.copy(), columns=['cluster'])
    df_cluster.reset_index(inplace=True)
    df_cluster.rename(columns={'index': 'idrefpelanggan'}, inplace=True)

    # return df_cluster

    # Get unique clusters
    unique_cluster = df_cluster['cluster'].unique()

    # Get ID ref based on cluster
    idrefs_list = []
    for i, x in enumerate(unique_cluster):
        idref_list = df_cluster.query("cluster == {}".format(x))[
            'idrefpelanggan'].values.tolist()
        #idrefs_list[x] = idref_list

        # Create dictionary
        idref_cluster_dict = {'cluster': x, 'idrefpelanggan': idref_list}
        idrefs_list.append(idref_cluster_dict)

    idrefs_cluster = pd.DataFrame(idrefs_list)
    return idrefs_cluster


seed = (0)
np.random.seed(seed)

# Convert data frame to list of series
pivoted_series = []
pivoted_columns = []
for i, y in enumerate(cols):
    length = len(df_pivoted1[y])
    cst = df_pivoted1[y].values
    pivoted_series.append(cst)
    pivoted_columns.append(y)


# Convert data set to standar time series format
formatted_dataset = to_time_series_dataset(pivoted_series)
print("Data shape: {}".format(formatted_dataset.shape))


# ### Data Normalization/Standarization


# Data normalization
formatted_norm_dataset = TimeSeriesScalerMeanVariance().fit_transform(formatted_dataset)
sz = formatted_norm_dataset.shape[1]
print("Data shape: {}".format(sz))

formatted_norm_dataset = TimeSeriesScalerMeanVariance().fit_transform(formatted_dataset)

totalColumn = formatted_norm_dataset.shape[0]
totalRow = formatted_norm_dataset.shape[1]
print(totalColumn)
print(totalRow)

clusters = 5
ks = KShape(n_clusters=clusters, verbose=True, random_state=seed)
y_pred_ks = ks.fit_predict(formatted_norm_dataset)


df_cluster = pd.DataFrame(
    y_pred_ks, index=pivoted_columns, columns=['cluster'])
df_cluster.reset_index(inplace=True)
df_cluster.rename(columns={'index': 'idrefpelanggan'}, inplace=True)
df_cluster.sort_values(['cluster'])


# Create data frame for customer and its cluster
create_cluster_info(y_pred_ks, cols)


plot_clusters(formatted_norm_dataset, y_pred_ks, clusters, ks,
              'pgn_customer_cluster_{}.jpg'.format(id_unit_usaha))


# Kmeans clustering with DBA-DTW distance metric
clusters = 5
dba_km = TimeSeriesKMeans(n_clusters=clusters,
                          metric="dtw",
                          max_iter_barycenter=20,
                          verbose=False,
                          random_state=seed)
y_pred_dbakm = dba_km.fit_predict(formatted_norm_dataset)

# Create data frame for customer and its cluster
create_cluster_info(y_pred_dbakm, cols)

# Plot cluster
plot_clusters(formatted_norm_dataset, y_pred_dbakm, clusters,
              dba_km, "./plot_custers_KMean_DBA_DTW.jpg")


# engine2 = sqlalchemy.create_engine(
#     'mssql+pyodbc://sa:ams123@192.168.5.216/SIPG?driver=SQL+Server')

# Session = sessionmaker(bind=engine2)
# session = Session()

# Base = declarative_base()


# class PL_CUSTOMER_CLUSTER(Base):
#     __tablename__ = 'PL_CUSTOMER_CLUSTER'

#     ID = Column(Integer, primary_key=True)
#     DATE_STAMP = Column(DateTime)
#     IDREFPELANGGAN = Column(String(30))
#     HOUR_NUM = Column(Integer)
#     CLUSTER_NUM = Column(Integer)
#     HOUR_NUM = Column(Integer)
#     FDVC_NORMALIZED = Column(Float)
#     AREA_ID = Column(String(5))


# for i in range(5):
#     print("cluster: " + str(i))
#     CLUSTER_NAME = "CENTROID_ID" + str(i)
#     cluster = i
#     for j in range(totalRow):
#         fdvc_norm = dba_km.cluster_centers_[i][j][0]
#         hour_num = j

#         sql = ""
#         item = PL_CUSTOMER_CLUSTER(DATE_STAMP=startDate, IDREFPELANGGAN=CLUSTER_NAME,
#                                    HOUR_NUM=hour_num, CLUSTER_NUM=cluster, FDVC_NORMALIZED=fdvc_norm, AREA_ID=id_unit_usaha)
#         session.add(item)
#         print("fdvc:" + str(fdvc_norm) + "Hour:" + str(hour_num))
#     # commit per id ref pelanngan
#     session.commit()
#     print(str(j) + ", " + str(fdvc_norm))


# # Kmeans clustering with soft-DTW distance metric
# clusters = 18
# sdtw_km = TimeSeriesKMeans(n_clusters=clusters,
#                            metric="softdtw",
#                            metric_params={"gamma": .01},
#                            verbose=True,
#                            random_state=seed)
# y_pred = sdtw_km.fit_predict(formatted_norm_dataset)


# plot_clusters(formatted_norm_dataset, y_pred, clusters,
#               sdtw_km, "./plot_custers_KMean_Soft_DTW.jpg")
