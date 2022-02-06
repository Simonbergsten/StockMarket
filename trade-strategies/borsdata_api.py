import requests
import pandas as pd
import time
import numpy as np
from pandas.tseries.offsets import MonthBegin, MonthEnd, Week
from Borsdata.constants import API_KEY
from scipy import stats



# pandas options for string representation of data frames (print)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option('display.width', 500)


def avoid_time_limit():
    time.sleep(0.51)



class Borsdata:

    def __init__(self):

        self._api_key = API_KEY
        self._url_root = "https://apiservice.borsdata.se/v1/"
        self._last_api_call = 0
        self._api_calls_per_second = 2
        self.stock_prices = None
        self.returns = None
        self.stock_information = self.get_instruments()
        self.returns_long = None
        self.prices_long = None

    requests.get
    def _call_api(self, url, params):
        """
        Internal function for API calls
        :param url: URL add to URL root
        :params: Additional URL parameters
        :return: JSON-encoded content, if any
        """

        current_time = time.time()
        time_delta = current_time - self._last_api_call

        if time_delta < 1 / self._api_calls_per_second:
            time.sleep(1 / self._api_calls_per_second - time_delta)

        response = requests.get(self._url_root + url, params)
        print(response.url)
        self._last_api_call = time.time()

        if response.status_code != 200:
            print(f"BorsdataAPI >> API-Error, status code: {response.status_code}")
            return response
        return response.json()

    def _set_index(self, df, index, ascending=True):
        """
        Set index(es) and sort by index
        :param df: pd.DataFrame
        :param index: Column name to set to index
        :param ascending: True to sort index ascending
        """
        if type(index) == list:
            for idx in index:
                if not idx in df.columns.array:
                    return
        else:
            if not index in df.columns:
                return

        df.set_index(index, inplace=True)
        df.sort_index(inplace=True, ascending=ascending)

    def _parse_date(self, df, key):
        """
        Parse date string as pd.datetime, if available
        :param df: pd.DataFrame
        :param key: Column name
        """
        if key in df:
            df[key] = pd.to_datetime(df[key])

    def _get_base_params(self):
        """
        Get URL parameter base
        :return: Parameters dict
        """
        return {
            "authKey": self._api_key,
            "version": 1,
            'maxYearCount': 20,
            'maxR12QCount': 40,
            'maxCount': 20
        }

    def _join_dataframes(self, frames):
        try:
            return pd.concat(frames, axis=1, sort=True).dropna()
        except ValueError:
            print("No index match criteria")

    def _generate_keys_from_company_name(self, df_instruments):
        """
        Skapar en hashsmap från namn på företag och insId. Detta för att kunna använda företagsnamn i sökning,
        istället för att hitta korrekt insId.
        :param: Dataframe med insId som index, Name/urlName/ticker som kolumn
        :return: Dictionary med name:insId
        """

        return pd.Series(df_instruments.index.values, index=df_instruments.name).to_dict()

    def get_instruments(self):

        url = "instruments"
        json_data = self._call_api(url, self._get_base_params())
        df = pd.json_normalize(json_data['instruments'])
        df = df[['insId', 'name', 'urlName', 'ticker', 'yahoo', 'sectorId',
                 'marketId', 'branchId', 'countryId', 'listingDate']]
        self._parse_date(df, "listingDate")
        self._set_index(df, 'insId')
        cols = ['sectorId', 'branchId']
        df[cols] = df[cols].dropna().applymap(np.int32)
        # df = self._sectors(df)
        df = self._replace_sectors(df)
        df = self._replace_countries(df)
        return df

    def get_countries(self):
        """
        Get country data
        :return: pd.DataFrame
        """
        url = "countries"
        json_data = self._call_api(url, self._get_base_params())
        df = pd.json_normalize(json_data["countries"])
        self._set_index(df, "id")
        return df

    def _replace_sectors(self, df):
        sectors = self._sectors_name()
        df2 = df.replace({"sectorId": sectors})

        return df2

    def _replace_countries(self, df):
        countries = self._countries_name()
        return df.replace({"countryId": countries})

    def _sectors_name(self):
        sectors = self.get_sectors()
        dict_ = pd.Series(sectors.name, index=sectors.index.values).to_dict()

        return dict_

    def _countries_name(self):
        countries = self.get_countries()
        return pd.Series(countries.name, index=countries.index.values).to_dict()

    def get_sectors(self):
        """
        Get sector data
        :return: pd.DataFrame
        """
        url = "sectors"
        json_data = self._call_api(url, self._get_base_params())
        df = pd.json_normalize(json_data["sectors"])
        self._set_index(df, "id")
        return df

    def filter_for_country(self, df, countries_to_keep):
        """Select companies from specific countries"""
        df2 = df[df['countryId'].isin([countries_to_keep])]
        self.stock_information = df2

    def filter_for_sector(self, df, sectors_to_keep):
        self.stock_information = df[df['sectorId'].isin([sectors_to_keep])]

    def get_instruments_updated(self):
        """
        Get all updated instruments
        :return: pd.DataFrame
        """
        url = "instruments/updated"
        json_data = self._call_api(url, self._get_base_params())
        df = pd.json_normalize(json_data["instruments"])
        self._parse_date(df, "updatedAt")
        self._set_index(df, "insId")
        return df

    def get_instrument_report(self, ins_id, report_type, max_count=None):
        """
        Get specific report data
        :param ins_id: Instrument ID
        :param report_type: ['quarter', 'year', 'r12']
        :param max_count: Max. number of history (quarters/years) to get
        :return: pd.DataFrame of report data
        """
        url = f"instruments/{ins_id}/reports/{report_type}"

        params = self._get_base_params()
        if max_count is not None:
            params["maxCount"] = max_count
        json_data = self._call_api(url, params)

        df = pd.json_normalize(json_data["reports"])
        df.columns = [x.replace("_", "") for x in df.columns]
        self._parse_date(df, "reportStartDate")
        self._parse_date(df, "reportEndDate")
        self._parse_date(df, "reportDate")
        self._set_index(df, ["year", "period"], ascending=False)
        return df

    def get_kpi_data_all_instruments(self, kpi_id, calc_group, calc, colname):
        """
        https://github.com/Borsdata-Sweden/API/wiki/KPI-Screener
        Get KPI data for all instruments
        :param kpi_id: KPI ID
        :param calc_group: ['last', '1year', '3year', '5year', '7year', '10year', '15year']
        :param calc: ['quarter', 'high', 'latest', 'mean', 'low', 'sum', 'cagr']
        :return: pd.DataFrame
        """
        url = f"instruments/kpis/{kpi_id}/{calc_group}/{calc}"
        json_data = self._call_api(url, self._get_base_params())
        df = pd.json_normalize(json_data["values"])
        df = df[['i', 'n']]
        df.rename(
            columns={"i": "insId", "n": colname},
            inplace=True,
        )
        self._set_index(df, "insId")
        self.stock_information = self.stock_information.join(df)

        # return df

    def get_instrument_stock_prices(self, df, from_=None, to=None, max_count=None):

        stockprices_list = []

        for idx, row in df.iterrows():
            url = f"instruments/{str(idx)}/stockprices"

            params = self._get_base_params()
            if from_ is not None:
                params["from"] = from_
            if to is not None:
                params["to"] = to
            if max_count is not None:
                params["maxCount"] = max_count
            json_data = self._call_api(url, params)

            df2 = pd.json_normalize(json_data["stockPricesList"])
            df2 = df2[["d", "c"]]
            df2.rename(columns={
                "d": "date",
                "c": row['urlName'] + "_close"  # idx,
            },
                inplace=True,
            )
            # df.rename(
            #     columns={
            #         "d": "date",
            #         "c": "close",
            #         "h": "high",
            #         "l": "low",
            #         "o": "open",
            #         "v": "volume",
            #     },
            #     inplace=True,
            # )
            self._parse_date(df2, "date")
            self._set_index(df2, "date", ascending=True)

            stockprices_list.append(df2)

        self.stock_prices = pd.concat(stockprices_list, axis=1)

    def calculate_returns(self, stockprices):
        returns = []
        # std = []
        # mean = []
        for index in stockprices.columns:
            returns.append(np.log(stockprices[index] / stockprices[index].shift(1)))
            # std.append(np.std(stockprices[index]) * np.sqrt(252))
            # mean.append(np.mean(stockprices[index] * 252))

        self.returns = pd.concat(returns, axis=1)
        self._wide_to_long(self.returns)

    def returns_correlation(self, returns):
        return returns.corr()


    def _wide_to_long(self, df):
        """Use this for pivoting so that we can calculate SMA"""
        df2 = df.reset_index()
        df2 = pd.melt(df2, id_vars = "date", var_name="stock", value_name="returns")
        df2['stock'] = df2.stock.str[0:-6]
        mask = df2.groupby(['stock'])['stock'].transform(self._mask_first).astype(bool)
        df2 = df2.loc[mask]
        self._parse_date(df2, "date")
        self._set_index(df2, "date", ascending=True)
        # self.returns_long = df2
        self.returns_long = df2.sort_values(by=["stock", "date"], ascending=[True, True])
        # return df2.sort_values(by = ["stock", "date"], ascending= [True, True])

    def _wide_to_long_stockprices(self, df):
        """Use this for pivoting stockprices to acheive the momentum portfolios."""
        df2 = df.reset_index()
        df2 = pd.melt(df2, id_vars="date", var_name="stock", value_name="close")
        df2['stock'] = df2.stock.str[0:-6]
        mask = df2.groupby(['stock'])['stock'].transform(self._mask_first).astype(bool)
        df2 = df2.loc[mask]
        self._parse_date(df2, "date")
        self._set_index(df2, "date", ascending=True)
        self.prices_long = df2.sort_values(by=["stock", "date"], ascending=[True, True])

    def rolling_window(self, window_length = 50):
        """Calculate rolling returns per stock"""
        # pd.concat([self.returns_long, returns_long.groupby('stock').rolling(window_length).mean()], axis=1)
        return self.returns_long.groupby('stock').rolling(window_length).mean()

    def _mask_first(self, x):
        result = np.ones_like(x)
        result[0] = 0
        return result

    def _clean_data(self, stock_prices, returns_type="discrete", J=3):
        """
        # Formatcrsp_m.set_indexLength: J can be betwecrsp_m.groupby months
        # Holding Period Length: K can be between 3 to 12 months
        """

        # Step 1: Reset index to monthtly data
        stock_prices = stock_prices.groupby("stock").resample("1M", closed="left").last()

        # # Ta bort index för stock name
        stock_prices = stock_prices.reset_index(level=0, drop=True)

        # Step 2: Gör en kolumn som heter date som är index
        # stock_prices["date"] = stock_prices.index
        stock_prices.reset_index(level = 0, inplace=True)

        stock_prices.sort_values(by=["stock", "date"], ascending=[True, True], inplace=True)

        # Step 3: Beräkna returns för respektive aktie
        if returns_type == "log":
            stock_prices["ret"] = stock_prices.groupby("stock")["close"].apply(lambda x: np.log(x) - np.log(x.shift()))
        else:
            stock_prices["ret"] = stock_prices.groupby("stock")["close"].apply(lambda x: x / x.shift() - 1)

        # # Step 4: Ta bort index
        # stock_prices = stock_prices.reset_index(level=0, drop=True)

        # Step 5 - Ta bort sista raden
        stock_prices = stock_prices[::-1]
        stock_prices.sort_values(by = ["stock", "date"], ascending=[True, True], inplace=True)

        # Step 6: Create tmp_crsp
        _tmp_stock_prices = stock_prices[["stock", "date", "ret"]].sort_values(['stock', 'date']).set_index("date")

        # Step 7:  Replace missing values with 0's
        _tmp_stock_prices["ret"] = _tmp_stock_prices["ret"].fillna(0)

        # Step 8:

        # Calculate rolling cumulative returns
        _tmp_stock_prices["logret"] = np.log(1 + _tmp_stock_prices["ret"])
        umd = _tmp_stock_prices.groupby(["stock"])["logret"].rolling(J, min_periods=J).sum()
        umd = umd.reset_index()
        umd["cumret"] = np.exp(umd["logret"]) - 1
        return stock_prices, umd, _tmp_stock_prices

    def formation_portfolios(self, K = 3, skipweek = False):
        stock_prices, umd, _tmp_stock_prices = api._clean_data(api.prices_long, returns_type= "log", J = 3)
        # For each date, assign ranking 1-10 based on cumret
        umd = umd.dropna(axis = 0, subset = ['cumret'])
        umd['momr'] = umd.groupby('date')['cumret'].transform(lambda x: pd.qcut(x.rank(method = "first"), 10, labels = False))
        # Ovanstående ger mig vilka aktier som har bäst respektive sämst performance under dessa perioder. Det vill säga,
        # detta skulle jag vilja plocka ut för att rangordna aktier baserat på performance
        umd.momr = umd.momr.astype(int)
        umd['momr'] = umd['momr'] + 1

        if skipweek == True:
            umd['hdate1'] = umd['date'] + MonthBegin(1) + Week(1)
        else:
            umd['hdate1'] = umd['date'] + MonthBegin(1)

        umd['hdate2'] = umd['date'] + MonthEnd(K)
        umd = umd.rename(columns = {'date':"form_date"})
        umd = umd[["stock", "form_date", "momr", "hdate1", "hdate2"]]

        # Här kan vi se vilka aktier som har högst/lägsta percentilen av aktierna.
        # umd.sort_values(by = ["form_date", "momr"] , ascending= [True, False], inplace=True)

        # Join rank and return data together
        _tmp_ret = stock_prices[['stock', 'date', 'ret']]
        port = pd.merge(_tmp_ret, umd, on = ['stock'], how = "inner")
        port = port[(port["hdate1"] <= port["date"]) & (port["date"] <= port["hdate2"])]

        umd2 = port.sort_values(by=["date", "momr", "form_date", "stock"]).drop_duplicates()
        umd3 = umd2.groupby(["date", "momr", "form_date"])["ret"].mean().reset_index()
        umd3.sort_values(by=["date", "momr"], ascending= [True, False], inplace=True)

        # Create one return series per MOM group each month
        ewret = umd3.groupby(['date', 'momr'])['ret'].mean().reset_index()
        ewstd = umd3.groupby(['date', 'momr'])['ret'].std().reset_index()
        ewret = ewret.rename(columns={'ret': 'ewret'})
        ewstd = ewstd.rename(columns={'ret': 'ewretstd'})
        ewretdat = pd.merge(ewret, ewstd, on=['date', 'momr'], how='inner')
        ewretdat = ewretdat.sort_values(by=['momr'])
        # ewretdat['ewretstd'] = ewretdat["ewretstd"].fillna(0)

        return ewretdat, umd

    def long_short_portfolios(self, ewretdat):
        # Transpose portfolio layout to have columns as portfolio returns
        ewretdat2 = ewretdat.pivot(index='date', columns='momr', values='ewret')

        # Add prefix port in front of each column
        ewretdat2 = ewretdat2.add_prefix('port')
        ewretdat2 = ewretdat2.rename(columns={'port1': 'losers', 'port10': 'winners'})
        ewretdat2['long_short'] = ewretdat2['winners'] - ewretdat2['losers']

        # Compute Long-Short Portfolio Cumulative Returns
        ewretdat3 = ewretdat2
        ewretdat3['1+losers'] = 1 + ewretdat3['losers']
        ewretdat3['1+winners'] = 1 + ewretdat3['winners']
        ewretdat3['1+ls'] = 1 + ewretdat3['long_short']

        ewretdat3['cumret_winners'] = ewretdat3['1+winners'].cumprod() - 1
        ewretdat3['cumret_losers'] = ewretdat3['1+losers'].cumprod() - 1
        ewretdat3['cumret_long_short'] = ewretdat3['1+ls'].cumprod() - 1

        return ewretdat2, ewretdat3

    # Summary statistics

    def get_stats(self, ewretdat3):

        # Mean
        mom_mean = ewretdat3[['winners', 'losers', 'long_short']].mean().to_frame()
        mom_mean = mom_mean.rename(columns={0: 'mean'}).reset_index()

        # T-Value and P-Value
        t_losers = pd.Series(stats.ttest_1samp(ewretdat3['losers'], 0.0)).to_frame().T
        t_winners = pd.Series(stats.ttest_1samp(ewretdat3['winners'], 0.0)).to_frame().T
        t_long_short = pd.Series(stats.ttest_1samp(ewretdat3['long_short'], 0.0)).to_frame().T

        t_losers['momr'] = 'losers'
        t_winners['momr'] = 'winners'
        t_long_short['momr'] = 'long_short'

        t_output = pd.concat([t_winners, t_losers, t_long_short]).rename(columns={0: 't-stat', 1: 'p-value'})

        # Combine mean, t and p
        mom_output = pd.merge(mom_mean, t_output, on=['momr'], how='inner')

        return mom_output


if __name__ == "__main__":
    api = Borsdata()
    print(api.get_instruments()[:20])
    # print(api.stock_information[:20])
    # api.filter_for_country(api.stock_information[:50], 'Sverige')
    # print(api.stock_information[:6])
    # print("\n")
    # print(api.get_instrument_report(3, "quarter"))
    # api.get_kpi_data_all_instruments("94", "1year", "mean", "rev_growth_last_year")
    # api.get_kpi_data_all_instruments("94", "last", "quarter", "rev_growth_last_quarter")
    # print(api.stock_information[:20])
    # api.get_instrument_stock_prices(api.stock_information[:100], from_="2019-01-01")
    # api.stock_prices
    # api.calculate_returns(api.stock_prices)
    # print(api.stock_prices)
    # api._wide_to_long_stockprices(api.stock_prices)
    # print(api.prices_long[:10])
    # print(api.prices_long)
    # x, y, z = api._clean_data(api.prices_long, returns_type="log", J=3)
    # print(x[:10])
    # print("\n")
    # print(y[:10])
    # print("\n")
    # print(z[:10])
    # ewretdat, umd = api.formation_portfolios()
    # ewretdat2, ewretdat3 = api.long_short_portfolios(ewretdat)
    # print(ewretdat3)
    # stats_ = api.get_stats(ewretdat3=ewretdat3)
    # print(stats_)

    # print(api.returns_correlation(api.returns))
    # print(api._wide_to_long(api.returns[:15]))
    # print(api.rolling_window(3)[:5])
    # print(api.returns_long[:5])
    # print(api.get_countries())
    # print("\n")
    # print(api.get_sectors())
    # print(ap.get)
#
# if __name__ == "__main__":
#     # Main, call functions here.
#     api = BorsdataAPI(constants.API_KEY)
#     # api.get_translation_meta_data()
#     # print(api.get_instruments_updated().head(10))
#     # print("\n")
#     # print(api.get_kpi_summary(3, "year").head(10))
#     print(api.get_kpi_data_instrument(3, 10, '1year', 'mean'))
#     print("1year mean")
#     # print(api.get_kpi_data_all_instruments(10, '1year', 'mean').dropna().head(25))
#     # api.get_updated_kpis()
#     # api.get_kpi_metadata()
#     # api.get_instrument_report(3, 'year')
#     # api.get_reports_metadata()
#     # api.get_stock_prices_date('2020-09-25')
#     # api.get_stock_splits()
#     print(api.get_countries())
