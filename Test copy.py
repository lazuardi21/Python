from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dateutil import parser
from collections import OrderedDict

s_year = ""
invStartDate = "2020-03-01"
invEndDate = "2020-09-02"
if s_year != "":
    invStartDate = str(s_year) + '-01-01'
    invEndDate = str(s_year) + '-12-01'
else:
    invStartDate = invStartDate
    invEndDate = invEndDate

print(invStartDate)
print(invEndDate)
