'''
Created on 2017-6-7

@author: Administrator
'''

from datetime import date, datetime, timedelta

path = ""
am = []
pm = []
with open("result\\inf_19_t585472\\result_all_inf.csv", "r") as f:
    all = f.read()
    lines = all.split('\n')
    j = 0
    for line in lines:
        if j % 12 in [0, 1, 2, 3, 4, 5]:
            am.append(line)
        else:
            pm.append(line)
        j = j + 1

with open(path + "finalResult" + ".csv", "w") as f:
    f.write(','.join(['"tollgate_id"', '"time_window"', '"direction"', '"volume"']) + '\n')
    index = 0
    for tol_dir in ["1_0", "1_1", "2_0", "3_0", "3_1"]:
        datei = datetime(2016, 10, 25)
        for i in range(7):
            start_window = datei + timedelta(hours=8)
            for i in range(6):
                stop_window = start_window + timedelta(minutes=20)
                strstart = start_window.strftime("%Y-%m-%d %H:%M:%S")
                strstop = stop_window.strftime("%Y-%m-%d %H:%M:%S")
                start_window = stop_window
                f.write(",".join(
                    [tol_dir.split("_")[0], "\"[" + strstart + "," + strstop + ")\"", tol_dir.split("_")[1],
                     am[index]]))
                index = index + 1
                f.write("\n")
        datei = datei + timedelta(days=1)
    index = 0
    for tol_dir in ["1_0", "1_1", "2_0", "3_0", "3_1"]:
        datei = datetime(2016, 10, 25)
        for i in range(7):
            start_window = datei + timedelta(hours=17)
            for i in range(6):
                stop_window = start_window + timedelta(minutes=20)
                strstart = start_window.strftime("%Y-%m-%d %H:%M:%S")
                strstop = stop_window.strftime("%Y-%m-%d %H:%M:%S")
                start_window = stop_window
                f.write(",".join(
                    [tol_dir.split("_")[0], "\"[" + strstart + "," + strstop + ")\"", tol_dir.split("_")[1], pm[index]]))
                f.write("\n")
                index = index + 1
        datei = datei + timedelta(days=1)
