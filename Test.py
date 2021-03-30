
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

startDate = "2020-01-01 00:00:00"
endDate = "2020-02-01 00:00:00"

login = ""
password = ""
# engine = sqlalchemy.create_engine('mysql+pymysql://energy:energy2x5=10@localhost:3306/pgn')
engine = sqlalchemy.create_engine(
    'mssql+pyodbc://sa:ams123@10.147.18.38/SIPG?driver=SQL+Server')

sql = "SELECT IDREFPELANGGAN, ID_UNIT_USAHA, FSTREAMID, DATEPART(dw,FDATETIME) as FDAYOFWEEK, FHOUR, avg(FDVC) as AVG_FDVC \
                FROM amr_bridge \
                WHERE FSTREAMID = 1 \
                AND FDVC > 0 \
                AND  FDATETIME >='" + startDate + "' and FDATETIME < '" + endDate + "'  \
                GROUP BY IDREFPELANGGAN, ID_UNIT_USAHA, FSTREAMID, DATEPART(dw,FDATETIME), FHOUR  "

df = pd.read_sql_query(sql, engine)
print(df)
