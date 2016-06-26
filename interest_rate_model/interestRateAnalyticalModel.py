import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import pylab as plt

tableNames = {
    "AUI":"Australia interest rate",
    "USI":"US interest rate daily",
    "AUE":"Australia exchange rates"
}

dbname = "interest_exchange_rate.db" # should include directory

pickle_name = "dt.pckl"

engine = create_engine("sqlite:///" + dbname)

def load_table(table_name):
    df = pd.read_sql_table(table_name,engine)
    df.set_index(df.columns[0],inplace=True)
    return df

def dump_pickle():
    dfAll = {key:load_table(tableNames[key]) for key in tableNames}
    aus_int_mon = dfAll["AUI"]["3-month BABs/NCDs"].interpolate()
    aus_int_mon.name = "AUI"
    us_int_day = dfAll['USI'].interpolate()
    us_int_mon = us_int_day.resample("M")
    us_int_mon.rename(columns={"0":"USI"},inplace=True)
    aus_ex_day = dfAll['AUE']["USD"].interpolate()
    aus_ex_mon = aus_ex_day.resample("M")
    aus_ex_mon.name = "AUD/USD"
    # cause we have more AUS monthly data then US
    aus_int_mon_cut = aus_int_mon[us_int_mon.index]
    aus_ex_mon_cut = aus_ex_mon[us_int_mon.index]
    
    dfToAnalyse = pd.concat([us_int_mon,aus_int_mon_cut,
                            aus_ex_mon_cut],axis=1)
    dfToAnalyse.dropna().to_pickle(pickle_name)

if __name__ == "__main__":
    # dump_pickle()
    df = pd.read_pickle(pickle_name)
    # df.plot(y="AUD/USD")
    # df.plot(y="USI")
    # df.plot(y="AUI")
    # plt.show()
    ex_one_month_advance = df['AUD/USD'][1:]
    ex_one_month_advance.index = df.index[:-1]
    ex_one_month_ratio = (ex_one_month_advance/df['AUD/USD']).dropna().values
    # print ex_one_month_ratio
    # int_ratio = (1 + 0.01*df['USI'][:-1]) / (1 + 0.01*df['AUI'][:-1])
    int_ratio = (-df['USI'][:-1] + df['AUI'][:-1]) * 0.01
    ex_vs_int = pd.DataFrame(int_ratio,columns=["int_ratio"])
    ex_vs_int["ex_one_month_ratio"] = np.log(ex_one_month_ratio)
    # ex_vs_int.plot(x="int_ratio",y="ex_one_month_ratio")
    ex_vs_int.plot()
    plt.show()
    # print ex_vs_int