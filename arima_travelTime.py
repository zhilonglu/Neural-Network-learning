from statsmodels.graphics.api import qqplot
from scipy import  stats
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import date,datetime,timedelta
path = 'C:\\Users\\zhilonglu\\Desktop\kdd\\trainData_lzl\\'
def process():
    file_dict ={}
    sort_file_dict={}
    with open(path+'training_20min_avg_travel_time.csv','r') as f1:
        f1.readline()  # skip the header
        data_1 = f1.readlines()
        f1.close()
    for i in range(len(data_1)):
        ls = data_1[i].replace('\"','').split(',')
        if ls[2].split(" ")[0] not in ["[2016-10-01","[2016-10-02","[2016-10-03","[2016-10-04","[2016-10-05","[2016-10-06","[2016-10-07"]:
            file_dict[(ls[0]+'_'+ls[1],ls[2]+','+ls[3])] = float(ls[4])
    # print file_dict
    sort_file_dict = sorted(file_dict.iteritems(),key=lambda d:d[0])
    # print sort_file_dict
    with open(path+'avg_volume_travel_time_dire.csv','w') as f:
        for i in sort_file_dict:
            # print i[0][0]
            # tollgate_dict[i[0][0]].append(i[1])
            f.write(','.join(i[0])+','+str(i[1])+'\n')
    dateTime=[]
    for h in ["06:","07:","08:","09:","15:","16:","17:","18:"]:
        for m in ["00:00","20:00","40:00"]:
            dateTime.append(h+m)
    # print dateTime
    forcast_time = []
    for h in ["08:","09:","17:","18:"]:
        for m in ["00:00","20:00","40:00"]:
            forcast_time.append(h+m)
    timeSerial_data={}
    for tol_dir in ["A_2","A_3","B_1","B_3","C_1","C_3"]:
        for time in forcast_time:
            timeSerial_data[(tol_dir,time)]=[]
    for i in sort_file_dict:
        key1 = i[0][0]
        key2 = i[0][1].split(',')[0].split(' ')[1]
        if key2 in forcast_time:
            timeSerial_data[(key1,key2)].append(i[1])
    # print timeSerial_data
    ar_model(timeSerial_data)

def ar_model(data):
    predict =["2016-10-18","2016-10-19","2016-10-20","2016-10-21","2016-10-22","2016-10-23","2016-10-24"]
    with open(path+'output_travel_time.csv','w') as f:
        for i in data:
            dta = data[i]
            dta=pd.Series(dta)
            start = '1760'
            end = str(int(1760)+len(dta)-1)
            dta.index = pd.Index(sm.tsa.datetools.dates_from_range(start,end))
            arma_mod20 = sm.tsa.AR(dta).fit()
            predict_start = str(int(end)+1)
            predict_end = str(int(predict_start)+6)
            predict_sunspots = arma_mod20.predict(predict_start,predict_end,dynamic=True)
            for j in range(len(predict_sunspots)):
                f.write(i[0]+","+predict[j]+" "+i[1]+","+str(predict_sunspots[j])+"\n")
            # fig, ax = plt.subplots(figsize=(12, 8))
            # ax = dta.ix['1760':].plot(ax=ax)
            # predict_sunspots.plot(ax=ax)
            # plt.show()
    f.close()
    file = path+'output_travel_time.csv'
    transTooutput(file)
def transTooutput(file):
    with open(file) as f1:
        data_1 = f1.readlines()
        f1.close()
        #1_0,2016-10-19 17:40:00,43.2624703991
    with open(path+'output_travel_time_final.csv','w') as f:
        f.write("intersection_id,tollgate_id,time_window,avg_travel_time\n")
        for i in range(len(data_1)):
            ls = data_1[i].split(",")
            datei=datetime.strptime(ls[1],"%Y-%m-%d %H:%M:%S")
            endi = datei+timedelta(minutes=20)
            endTime=endi.strftime("%Y-%m-%d %H:%M:%S")
            intersection_id = ls[0].split("_")[0]
            tollgate_id = ls[0].split("_")[1]
            f.write(str(intersection_id)+","+str(tollgate_id)+",\"["+ls[1]+","+endTime+")\""+","+str(ls[2]))
def main():
    process()

if __name__ == '__main__':
    main()
