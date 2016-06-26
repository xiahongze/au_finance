import pandas as pd

__author__ = "Hongze XIA"
__date__ = "26 Jun 2016"

fileListedCompanies = "ASXListedCompanies.pckl"
stockAXS = pd.read_pickle(fileListedCompanies)
stockAXS.set_index("ASX code",inplace=True)

# a,b,c,d,e,f = code,startMonth (jan=00),day,year,endMonth,day,year
urlYahooFinance = "http://real-chart.finance.yahoo.com/table.csv?s=" +\
                    "{:s}.AX&a={:02d}&b={:02d}&c={:4d}&" +\
                    "d={:02d}&e={:02d}&f={:4d}&g=d&ignore=.csv"

def get_histdata(symbol,startDate="1990-01-01",endDate=None):
    """
    symbol: string
        AXS code for a stock
    startDate,endDate: date string or datetime object
        this can be neglected
    """
    startDate = pd.to_datetime(startDate)
    if endDate is not None:
        endDate = pd.to_datetime(endDate)
    else:
        endDate = pd.datetime.now()
    url = urlYahooFinance.format(
                symbol,startDate.month-1,startDate.day,startDate.year,
                endDate.month-1,endDate.day,endDate.year
        )
    # print url
    
    try:
        df = pd.read_csv(url)
        df.set_index("Date",inplace=True)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df
    except:
        print "Couldn't find this stock at Yahoo Finance or banned by Yahoo somehow!"
        return None

def get_selected_hist(stocks):
    """
    stocks: list of string,integer or just str and int
        a list of AXS stock codes/symbols
        if None, return all of the ASX stocks
    return:
        a dictionary
    """
    if type(stocks) is list:
        if type(stocks[0]) is str:
            alist = stocks
        elif type(stocks[0]) is int:
            alist = stockAXS.index[stocks]
    elif type(stocks) is str:
        alist = [stocks]
    elif type(stocks) is int:
        alist = [stockAXS.index[stocks]]
    elif stocks is None:
        alist = stockAXS.index
    
    return {s:get_histdata(s) for s in alist}

def print_stock_info(stocks):
    print stockAXS.loc[stocks]

#################### TEST ##################
# df = get_histdata("TOT")
# print df.head()
# d = get_selected_hist(['ACB','AAU'])
# print d
# print_stock_info(['AAU','ADD'])