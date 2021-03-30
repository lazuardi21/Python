from operator import inv
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from tslearn.clustering import TimeSeriesKMeans, KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax, TimeSeriesResampler
from tslearn.utils import to_time_series_dataset
from sqlalchemy import create_engine
from IPython import get_ipython
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil import parser
from datetime import datetime, timedelta
from collections import OrderedDict

import pandas as pd
import pycaret
import sqlalchemy
import numpy as np
import matplotlib.pyplot as plt
import time

plt.style.use('seaborn')

# single input

# startDate = "2021-01-01 00:00:00"
# startDate = datetime.strptime(startDate, '%Y-%m-%d %H:%M:%S')
# endDate = startDate + relativedelta(months=1)
# print(endDate)

# for input in single month
startDate = ""
endDate = ""

# input with interval year-month-date
s_year = ""
invStartDate = "2020-03-01"
invEndDate = "2020-04-01"
id_unit_usaha = '017'
if invEndDate != "":
    invEndDate = datetime.strptime(invEndDate, '%Y-%m-%d')
    invEndDate = invEndDate - relativedelta(months=1)
    # invEndDate = str(invEndDate)
    invEndDate = invEndDate.strftime("%Y-%m-%d")
    # invEndDate = datetime.strptime(invEndDate, '%Y-%m-%d')
    invEndDate = str(invEndDate)
else:
    invEndDate = invEndDate


id_unit_usaha = [id_unit_usaha]

if s_year != "":
    invStartDate = str(s_year) + '-01-01'
    invEndDate = str(s_year) + '-12-01'
else:
    invStartDate = invStartDate
    invEndDate = invEndDate

if id_unit_usaha[0] == 'All':
    id_unit_usaha2 = ['011', '012', '013', '014', '015', '016',
                      '017', '018', '019', '021', '022', '023', '024', '031', '033']
else:
    id_unit_usaha2 = id_unit_usaha

dates = [invStartDate, invEndDate]


def date(dates, id_unit_usaha):
    start, end = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]
    def total_months(dt): return dt.month + 12 * dt.year
    mlist = []
    for tot_m in range(total_months(start)-1, total_months(end)+1):
        y, m = divmod(tot_m, 12)
        mlist.append(datetime(y, m+1, 1).strftime("%Y-%m-%d %H:%M:%S"))
    # print(mlist)
    z = date_list(mlist, id_unit_usaha)
    return z


def date_list(mlist, id_unit_usaha):
    w = 0
    for x in range(len(mlist)-1):
        startDate = mlist[x]
        endDate = mlist[x+1]
        print('startDate: ' + startDate + ' endDate: ' + endDate)

        v = mass_upload(startDate, endDate, id_unit_usaha)
        w = v + w
    return w


def standard_Date(id_unit_usaha):
    interval_date = relativedelta(months=1)
    date_after_month = datetime.today() + relativedelta(months=1)
    startDate = datetime.today().strftime('%Y-%m-%d %H:%M:%S:%f')
    startDate = datetime.strptime(startDate, '%Y-%m-%d %H:%M:%S:%f')
    # endDate = date_after_month.strftime('%Y-%m-%d %H:%M:%S')
    endDate = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    endDate = datetime.strptime(endDate, '%Y-%m-%d %H:%M:%S')

    a = startDate
    b = startDate.year
    c = startDate.month
    d = startDate.day
    e = startDate.hour
    f = startDate.minute
    g = startDate.second
    h = startDate.microsecond
    i = str(0)

    if c > 1:
        c = c
    else:
        c = c

    if d > 1:
        d = '01'
    else:
        d = d

    if e > 1:
        e = '00'
    else:
        e = e

    if f > 1:
        f = '00'
    else:
        f = f

    if g > 1:
        g = '00'
    else:
        g = g

    if h > 1:
        h = '000000'
    else:
        h = h

    a = str(a)
    b = str(b)
    c = i + str(c)
    d = str(d)
    e = str(e)
    f = str(f)
    g = str(g)
    h = str(h)

    endDate = b + '-' + c + '-' + d + ' ' + e + ':' + f + ':' + g + '.' + h
    startDate = parser.parse(endDate) - relativedelta(months=1)
    endDate = parser.parse(endDate) - relativedelta(months=1)

    startDate = startDate.strftime("%Y-%m-%d")
    endDate = endDate.strftime("%Y-%m-%d")
    startDate = str(startDate)
    endDate = str(endDate)
    print(startDate, ' ', endDate)
    dates = [startDate, endDate]
    l = date(dates, id_unit_usaha)
    return l


def mass_upload(startDate, endDate, id_unit_usaha):
    print(id_unit_usaha)
    login = ""
    password = ""
    # engine = sqlalchemy.create_engine('mysql+pymysql://energy:energy2x5=10@localhost:3306/pgn')
    engine = sqlalchemy.create_engine(
        'mssql+pyodbc://sa:ams123@192.168.5.216/SIPG?driver=SQL+Server')

    sql = " SELECT a.IDREFPELANGGAN, a.ID_UNIT_USAHA, 1 AS FSTREAMID, DATEPART(dw, a.FDATETIME) as FDAYOFWEEK, a.FHOUR, AVG(a.FDVC) as AVG_FDVC\
            FROM(SELECT IDREFPELANGGAN, ID_UNIT_USAHA, FDATETIME, FHOUR, SUM(FDVC) as FDVC\
                FROM amr_bridge\
                WHERE FDATETIME >= '" + startDate + "'\
                and FDATETIME < '" + endDate + "'\
                GROUP BY IDREFPELANGGAN, ID_UNIT_USAHA, FDATETIME, FHOUR) a\
            GROUP BY a.IDREFPELANGGAN, a.ID_UNIT_USAHA, DATEPART(dw, a.FDATETIME), a.FHOUR\
            ORDER BY a.IDREFPELANGGAN, a.ID_UNIT_USAHA, DATEPART(dw, a.FDATETIME), a.FHOUR"

    df = pd.read_sql_query(sql, engine)
    totaldf = len(df)
    totaldf = str(totaldf)
    print('total Data: ' + totaldf)
    # rslt_df = df.loc[df['ID_UNIT_USAHA'] == '014']

    # print(startDate)
    # print('\nResult dataframe :\n', rslt_df)

    # df.to_csv('pgn_customer_cluster_v1_{}.csv'.format(id_unit_usaha), index=False)

    # df.to_hdf("amr_bridge_22122020.hdf", key='hdf5')

    # df = pd.read_hdf("amr_bridge_22122020.hdf")

    def select_data(id_unit):
        query = "ID_UNIT_USAHA == '{}'".format(id_unit_usaha)
        columns = ['FDAYOFWEEK', 'FHOUR', 'IDREFPELANGGAN', 'AVG_FDVC']

        # df = df.set_index('FDATETIME')
        df_selected = df.query(query, engine='python')[columns]
        return df_selected

    def pivot_data(df):
        # df_pivoted = df.pivot(index='FDATETIME', columns='IDREFPELANGGAN', values='FDVC')
        df_pivoted = df.pivot(
            index=['FDAYOFWEEK', 'FHOUR'], columns='IDREFPELANGGAN', values='AVG_FDVC')
        return df_pivoted

    def remove_zerocolumns(df):
        # Get all columns which have all zero values
        cols = df.columns[df.mean() == 0]
        # Drop columns which has all zero values
        df = df.drop(cols, axis=1)
        return df

    df_week1 = select_data(id_unit_usaha)
    df_week1.fillna(0.0, inplace=True)

    df_pivoted1 = pivot_data(df_week1)
    df_pivoted1.fillna(0.0, inplace=True)

    df_pivoted1 = remove_zerocolumns(df_pivoted1)
    cols = list(df_pivoted1.columns)
    df_pivoted1.head()

    # Function to plot cluster

    # def plot_clusters(ds, y_pred, n_clusters, ks, filename):
    #     plt.figure(figsize=(12, 40))
    #     for yi in range(n_clusters):
    #         plt.subplot(n_clusters, 1, 1 + yi)
    #         for xx in ds[y_pred == yi]:
    #             plt.plot(xx.ravel(), "k-", alpha=.2)
    #         plt.plot(ks.cluster_centers_[yi].ravel(), "r-")
    #         plt.xlim(0, sz)
    #         plt.ylim(-7, 7)
    #         plt.title("Cluster %d" % (yi))

    #     plt.tight_layout()
    #     plt.savefig(filename, format='jpg', dpi=300, quality=95)
    #     plt.show()

    def create_cluster_info(y_pred, cols):

        df_cluster = pd.DataFrame(
            y_pred.copy(), index=cols.copy(), columns=['cluster'])
        df_cluster.reset_index(inplace=True)
        df_cluster.rename(
            columns={'index': 'idrefpelanggan'}, inplace=True)

        unique_cluster = df_cluster['cluster'].unique()

        # Get ID ref based on cluster
        idrefs_list = []
        for i, x in enumerate(unique_cluster):
            idref_list = df_cluster.query("cluster == {}".format(x))[
                'idrefpelanggan'].values.tolist()
            # idrefs_list[x] = idref_list

            # Create dictionary
            idref_cluster_dict = {'cluster': x,
                                  'idrefpelanggan': idref_list}
            idrefs_list.append(idref_cluster_dict)

        idrefs_cluster = pd.DataFrame(idrefs_list)
        return idrefs_cluster

    # def run_once(startime, totalData, _has_run=[]):
    #     if _has_run:
    #         return
    #     # print("run_once doing stuff")
    #     print(startime)
    #     endtime = time.time_ns()
    #     print(endtime)
    #     invTime = endtime-startime

    #     estTime = invTime * totalData
    #     _has_run.append(1)

    #     print(totalData)
    #     print(estTime)
    #     return estTime

    seed = 0
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

    formatted_norm_dataset = TimeSeriesScalerMeanVariance().fit_transform(formatted_dataset)
    sz = formatted_norm_dataset.shape[1]
    print("Data shape: {}".format(sz))

    formatted_norm_dataset = TimeSeriesScalerMeanVariance().fit_transform(formatted_dataset)
    clusters = 5
    totalColumn = formatted_norm_dataset.shape[0]
    totalRow = formatted_norm_dataset.shape[1]
    totalData = totalRow*totalColumn + totalRow*clusters

    ks = KShape(n_clusters=clusters, verbose=True, random_state=seed)
    y_pred_ks = ks.fit_predict(formatted_norm_dataset)
    formatted_norm_dataset.shape
    data = formatted_norm_dataset
    data.shape

    formatted_norm_dataset_2d = formatted_norm_dataset[:, :, 0]
    formatted_norm_dataset_2d.shape
    # pd.DataFrame(A.T.reshape(2, -1), columns=cols)

    df_normalized = pd.DataFrame(formatted_norm_dataset_2d)
    df_normalized
    # df_normalized = df_normalized.pivot()
    # formatted_norm_dataset[0]

    df_cluster = pd.DataFrame(
        y_pred_ks, index=pivoted_columns, columns=['cluster'])
    df_cluster.reset_index(inplace=True)
    df_cluster.rename(columns={'index': 'idrefpelanggan'}, inplace=True)
    df_cluster.sort_values(['cluster'])

    df_normalized_detail = pd.DataFrame.join(df_normalized, df_cluster)
    df_normalized_detail

    # df_cluster.to_csv('pgn_customer_cluster_{}.csv'.format(
    #     id_unit_usaha), index=False)

    # Create data frame for customer and its cluster
    create_cluster_info(y_pred_ks, cols)

    # plot_clusters(formatted_norm_dataset, y_pred_ks, clusters, ks,
    #               'pgn_customer_cluster_{}.jpg'.format(id_unit_usaha))

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
    # startime = time.time_ns()
    # for i in range(totalColumn):

    #     idref = df_normalized_detail.iloc[i, totalRow]
    #     cluster = int(df_normalized_detail.iloc[i, totalRow+1])
    #     print("idref = " + idref)
    #     cluster_num = df_normalized_detail.iloc[i, totalRow-1]
    #     for j in range(totalRow):

    #         hour_num = df_normalized_detail.columns[j]
    #         fdvc = df_normalized_detail.iloc[i, j]

    #         sql = ""

    #         # insert into table
    #         item = PL_CUSTOMER_CLUSTER(DATE_STAMP=startDate, IDREFPELANGGAN=idref,
    #                                    HOUR_NUM=hour_num, CLUSTER_NUM=cluster, FDVC_NORMALIZED=fdvc, AREA_ID=id_unit_usaha)
    #         session.add(item)

    #     # commit per id ref pelanngan
    #     session.commit()

    engine2 = sqlalchemy.create_engine(
        'mssql+pyodbc://sa:ams123@192.168.5.216/SIPG?driver=SQL+Server')

    Session = sessionmaker(bind=engine2)
    session = Session()

    Base = declarative_base()

    class PL_CUSTOMER_CLUSTER(Base):
        __tablename__ = 'PL_CUSTOMER_CLUSTER'

        ID = Column(Integer, primary_key=True)
        DATE_STAMP = Column(DateTime)
        IDREFPELANGGAN = Column(String(30))
        HOUR_NUM = Column(Integer)
        CLUSTER_NUM = Column(Integer)
        HOUR_NUM = Column(Integer)
        FDVC_NORMALIZED = Column(Float)
        AREA_ID = Column(String(5))

    df_normalized_detail

    for i in range(5):
        print("cluster: " + str(i))
        CLUSTER_NAME = "CENTROID_ID" + str(i)
        cluster = i
        for j in range(totalRow):
            fdvc_norm = ks.cluster_centers_[i][j][0]
            hour_num = j

            sql = ""
            item = PL_CUSTOMER_CLUSTER(DATE_STAMP=startDate, IDREFPELANGGAN=CLUSTER_NAME,
                                       HOUR_NUM=hour_num, CLUSTER_NUM=cluster, FDVC_NORMALIZED=fdvc_norm, AREA_ID=id_unit_usaha)
            session.add(item)
            print("fdvc:" + str(fdvc_norm) + "Hour:" + str(hour_num))
        # commit per id ref pelanngan
        session.commit()
        print(str(j) + ", " + str(fdvc_norm))

    return totalData


if startDate != "" and endDate != "":
    startDate = startDate
    endDate = endDate
    print(startDate)
    print(endDate)
    for i in range(len(id_unit_usaha2)):
        id_unit_usaha = id_unit_usaha2[i]
        mass_upload(startDate, endDate, id_unit_usaha)
elif invStartDate != "" and invEndDate != "":
    for i in range(len(id_unit_usaha2)):
        id_unit_usaha = id_unit_usaha2[i]
        k = date(dates, id_unit_usaha)
        print(k)
else:
    for i in range(len(id_unit_usaha2)):
        id_unit_usaha = id_unit_usaha2[i]
        m = standard_Date(id_unit_usaha)
        print(m)

# startDate = "2021-02-01 00:00:00"
# endDate = "2021-03-01 00:00:00"
# endDate = str(endDate)
