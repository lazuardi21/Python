import tkinter as tk
from datetime import datetime
from dateutil.relativedelta import relativedelta

root = tk.Tk()

interval_date = relativedelta(months=1)
date_after_month = (datetime.today() + relativedelta(months=1))
# startdate = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
enddate = date_after_month.strftime('%Y-%m-%d %H:%M:%S')

startDate = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
startDate = datetime.strptime(startDate, '%Y-%m-%d %H:%M:%S')

a = str(startDate)
b = str(enddate)
# c = str(startdate)


print(startDate)
print(enddate)
print(a)
print(b)
# print(c)
# enddate = startdate + interval_date
# endDate =
# print("Hello World " + a)

canvas1 = tk.Canvas(root, width=500, height=500)
canvas1.pack()

label1 = tk.Label(root, text='Hello World! ' + a + ' ' + b)
canvas1.create_window(250, 250, window=label1)

#  +startdate + 'date:' + date_after_month

root.mainloop()
