import pandas as pd
import glob
import numpy as np

def grab_xls_as_df(filename):
    print filename
    xls = pd.ExcelFile(filename)
    df = xls.parse(index_col=0,skiprows=1)
    df.index.name = "Date"
    df.drop(df.index[:9],inplace=True)
    df.index = df.index.to_datetime()
    df.columns = map(lambda s: s[4:] if s[0] == 'A' else "TWI1970",df.columns)
    # df.replace(to_replace="Closed",value=np.nan,inplace=True)
    # df.replace(to_replace="CLOSED",value=np.nan,inplace=True)
    # df.replace(to_replace=' --',value=np.nan,inplace=True)
    df.replace(to_replace=r'[-\w]',value=np.nan,inplace=True,regex=True)
    df = df.astype(np.float32)
    return df

if __name__ == "__main__":
    files = glob.glob("*.xls")
    dfList = map(grab_xls_as_df,files)
    # df = grab_xls_as_df(files[0])
    dfAll = pd.concat(dfList)
    print dfAll
    # dbfile = '../finance.db'
    # tbname = 'Australia exchange rates'
    # import sqlite3
    # conn = sqlite3.connect(dbfile)
    # dfAll.to_sql(name=tbname,con=conn)
    # conn.close()