from __future__ import annotations

import alpaca_trade_api as tradeapi
import exchange_calendars as tc
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from models.fnn import fnn_prediction
from processing.indicators import process_indicators
from utils.utils import get_var

API_KEY = get_var("ALPACA_API_KEY")
API_SECRET = get_var("ALPACA_API_SECRET")
API_BASE_URL = get_var("ALPACA_API_BASE_URL")


# reference: https://github.com/AI4Finance-Foundation/FinRL

class AlpacaProcessor:
    def __init__(
            self, 
            API_KEY=API_KEY, 
            API_SECRET=API_SECRET, 
            API_BASE_URL=API_BASE_URL, 
            api=None,
            save_scaler=False,
            time_interval='1Min',
            reset: bool=True
    ):
        if api is None:
            try:
                self.api = tradeapi.REST(
                    API_KEY, API_SECRET, API_BASE_URL, "v2"
                )
            except BaseException:
                raise ValueError("Wrong Account Info!")
        else:
            self.api = api
        
        self.save_scaler = save_scaler
        self.time_interval = time_interval

        if reset:
            if hasattr(self, 'start'):
                del self.start
            if hasattr(self, 'end'):
                del self.end

    def _fetch_data_for_ticker(
            self, 
            ticker, 
            start_date, 
            end_date, 
            time_interval
    ):
        bars = self.api.get_bars(
            ticker,
            time_interval,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
        ).df
        bars["symbol"] = ticker
        return bars

    def download_data(
        self, ticker_list, start_date, end_date, time_interval
    ) -> pd.DataFrame:
        """
        Downloads data using Alpaca's tradeapi.REST method.

        Parameters:
        - ticker_list : list of strings, each string is a ticker
        - start_date : string in the format 'YYYY-MM-DD'
        - end_date : string in the format 'YYYY-MM-DD'
        - time_interval: string representing the interval ('1D', '1Min', etc.)

        Returns:
        - pd.DataFrame with the requested data
        """
        self.start = start_date
        self.end = end_date
        self.time_interval = time_interval 

        NY = "America/New_York"
        start_date = pd.Timestamp(start_date + " 09:30:00", tz=NY)
        end_date = pd.Timestamp(end_date + " 15:59:00", tz=NY)

        # Use ThreadPoolExecutor to fetch data for 
        # multiple tickers concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(
                    self._fetch_data_for_ticker,
                    ticker,
                    start_date,
                    end_date,
                    time_interval,
                )
                for ticker in ticker_list
            ]
            data_list = [future.result() for future in futures]

        # Combine the data
        data_df = pd.concat(data_list, axis=0)

        # Convert the timezone
        data_df = data_df.tz_convert(NY)

        # If time_interval is less than a day, 
        # filter out the times outside of NYSE trading hours
        if pd.Timedelta(time_interval) < pd.Timedelta(days=1):
            data_df = data_df.between_time("09:30", "15:59")

        # Reset the index and rename the columns for consistency
        data_df = data_df.reset_index().rename(
            columns={
                "index": "timestamp", 
                "symbol": "tic",
            }
        )

        # Sort the data by both timestamp and tic for consistent ordering
        data_df = data_df.sort_values(by=["tic", "timestamp"])

        # Reset the index and drop the old index column
        data_df = data_df.reset_index(drop=True)

        return data_df

    @staticmethod
    def clean_individual_ticker(args):
        tic, df, times = args
        tmp_df = pd.DataFrame(index=times)
        tic_df = df[df.tic == tic].set_index("timestamp")

        # Step 1: Merging dataframes to avoid loop
        tmp_df = tmp_df.merge(
            tic_df[["open", "high", "low", "close", "volume"]],
            left_index=True,
            right_index=True,
            how="left",
        )

        # Step 2: Handling NaN values efficiently
        if pd.isna(tmp_df.iloc[0]["close"]):
            first_valid_index = tmp_df["close"].first_valid_index()
            if first_valid_index is not None:
                first_valid_price = tmp_df.loc[first_valid_index, "close"]
                print(
                    f"The price of the first row for ticker {tic} is NaN. "
                    "It will be filled with the first valid price."
                )
                tmp_df.iloc[0] = [first_valid_price] * 4 + [0.0]  # Set volume to zero
            else:
                print(
                    f"Missing data for ticker: {tic}. The prices are all NaN. Fill with 0."
                )
                tmp_df.iloc[0] = [0.0] * 5

        for i in range(1, tmp_df.shape[0]):
            if pd.isna(tmp_df.iloc[i]["close"]):
                previous_close = tmp_df.iloc[i - 1]["close"]
                tmp_df.iloc[i] = [previous_close] * 4 + [0.0]

        # Setting the volume for the market opening timestamp to zero - Not needed
        # tmp_df.loc[tmp_df.index.time == pd.Timestamp("09:30:00").time(), 'volume'] = 0.0

        # Step 3: Data type conversion
        tmp_df = tmp_df.astype(float)

        tmp_df["tic"] = tic

        return tmp_df

    def clean_data(self, df):
        # this captures data that is stored in s3
        df = df.rename(
            columns={
                "ticker": "tic"
            }
        )
        df = df[['timestamp','open','high','low','close','volume','tic']]

        # if dates are none when cleaning, then we need to set
        if not hasattr(self, 'start'):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Find the earliest and latest dates in the DataFrame
            self.start = df['timestamp'].min().strftime('%Y-%m-%d')
            self.end = df['timestamp'].max().strftime('%Y-%m-%d')

        print("Data cleaning started")
        tic_list = np.unique(df.tic.values)
        n_tickers = len(tic_list)

        print("align start and end dates")
        grouped = df.groupby("timestamp")
        filter_mask = grouped.transform("count")["tic"] >= n_tickers
        df = df[filter_mask]

        trading_days = self.get_trading_days(start=self.start, end=self.end)

        # produce full timestamp index
        print("produce full timestamp index")
        times = []
        for day in trading_days:
            NY = "America/New_York"
            current_time = pd.Timestamp(day + " 09:30:00").tz_localize(NY)
            for i in range(390):
                times.append(current_time)
                current_time += pd.Timedelta(minutes=1)

        print("Start processing tickers")

        future_results = []
        for tic in tic_list:
            result = self.clean_individual_ticker((tic, df.copy(), times))
            future_results.append(result)

        print("ticker list complete")

        print("Start concat and rename")
        new_df = pd.concat(future_results)
        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "timestamp"})

        print("Data clean finished!")

        return new_df

    def add_technical_indicators(
        self, df, add_fnn: bool=True
    ) -> pd.DataFrame():
        print("Started adding Indicators")
        # Store the original data type of the 'timestamp' column
        original_timestamp_dtype = df["timestamp"].dtype

        unique_tickers = df.tic.unique()

        indicator_df = pd.DataFrame()
        for u_tic in unique_tickers:
            tic_df = df[df.tic == u_tic]
            tic_df = process_indicators(tic_df)
            indicator_df = pd.concat(
                [indicator_df, tic_df], ignore_index=True
            )

        print("Restore Timestamps")
        # Restore 'timestamp' column original data type
        if isinstance(original_timestamp_dtype, pd.DatetimeTZDtype):
            if indicator_df["timestamp"].dt.tz is None:
                indicator_df["timestamp"] = \
                    indicator_df["timestamp"].dt.tz_localize("UTC")
            indicator_df["timestamp"] = \
                indicator_df["timestamp"].dt.tz_convert(
                    original_timestamp_dtype.tz)
        else:
            indicator_df["timestamp"] = \
                indicator_df["timestamp"].astype(original_timestamp_dtype)
        
        # adding FNN to indicators
        if add_fnn:
            indicator_df = fnn_prediction(indicator_df)

        print("Finished adding Indicators")
        return indicator_df
    
    def preprocess_data(
            self,
            df: pd.DataFrame, 
        ) -> pd.DataFrame:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # exclude target var close price
        columns_to_exclude = ['close']
        numeric_cols = df.select_dtypes(include=[np.number]) \
            .drop(columns=columns_to_exclude)

        numeric_cols_transformed = numeric_cols.apply(np.arcsinh)

        df.update(numeric_cols_transformed)

        def scale_numeric_columns(group):
            numeric_cols = group.select_dtypes(include=[np.number]) \
                .drop(columns=columns_to_exclude)
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(numeric_cols)
            group.loc[:, numeric_cols.columns] = scaled_values
            return group

        df = df.groupby(df['timestamp'].dt.date) \
            .apply(scale_numeric_columns)
        df.reset_index(drop=True, inplace=True)

        return df

    # Allows to multithread the add_vix function for quicker execution
    def download_and_clean_data(self):
        vix_df = self.download_data(
            ["VIXY"], self.start, self.end, self.time_interval
        )
        return self.clean_data(vix_df)

    def add_vix(self, data):
        with ThreadPoolExecutor() as executor:
            future = executor.submit(self.download_and_clean_data)
            cleaned_vix = future.result()

        vix = cleaned_vix[["timestamp", "close"]]

        merge_column = "date" if "date" in data.columns else "timestamp"

        vix = vix.rename(
            columns={"timestamp": merge_column, "close": "VIXY"}
        )  # Change column name dynamically

        data = data.copy()
        data = data.merge(
            vix, on=merge_column
        )  # Use the dynamic column name for merging
        data = data.sort_values([merge_column, "tic"]).reset_index(drop=True)

        return data

    def calculate_turbulence(self, data, time_period=252):
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(
            index="timestamp", columns="tic", values="close"
        )
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.timestamp.unique()
        # start after a fixed timestamp period
        start = time_period
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - time_period])
            ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[
                hist_price.isna().sum().min() :
            ].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame(
            {"timestamp": df_price_pivot.index, "turbulence": turbulence_index}
        )

        # print("turbulence_index\n", turbulence_index)

        return turbulence_index

    def add_turbulence(self, data, time_period=252):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df, time_period=time_period)
        df = df.merge(turbulence_index, on="timestamp")
        df = df.sort_values(["timestamp", "tic"]).reset_index(drop=True)
        return df

    def df_to_array(self, df, tech_indicator_list, if_vix):
        df = df.copy()
        unique_ticker = df.tic.unique()
        if_first_time = True
        for tic in unique_ticker:
            if if_first_time:
                price_array = df[df.tic == tic][["close"]].values
                tech_array = df[df.tic == tic][tech_indicator_list].values
                if if_vix:
                    turbulence_array = df[df.tic == tic]["VIXY"].values
                else:
                    turbulence_array = df[df.tic == tic]["turbulence"].values
                if_first_time = False
            else:
                price_array = np.hstack(
                    [price_array, df[df.tic == tic][["close"]].values]
                )
                tech_array = np.hstack(
                    [tech_array, df[df.tic == tic][tech_indicator_list].values]
                )

        # Fill nan and inf values with 0 for technical indicators
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0
        tech_inf_positions = np.isinf(tech_array)
        tech_array[tech_inf_positions] = 0

        return price_array, tech_array, turbulence_array

    def get_trading_days(self, start, end):
        nyse = tc.get_calendar("NYSE")
        df = nyse.sessions_in_range(
            pd.Timestamp(
                start).tz_localize(None), pd.Timestamp(end).tz_localize(None
            )
        )
        trading_days = []
        for day in df:
            trading_days.append(str(day)[:10])

        return trading_days

    def fetch_latest_data(
        self, ticker_list, time_interval, tech_indicator_list, limit=100
    ) -> pd.DataFrame:
        data_df = pd.DataFrame()
        for tic in ticker_list:
            barset = self.api.get_bars([tic], time_interval, limit=limit).df  # [tic]
            barset["tic"] = tic
            barset = barset.reset_index()
            data_df = pd.concat([data_df, barset])

        data_df = data_df.reset_index(drop=True)
        start_time = data_df.timestamp.min()
        end_time = data_df.timestamp.max()
        times = []
        current_time = start_time
        end = end_time + pd.Timedelta(minutes=1)
        while current_time != end:
            times.append(current_time)
            current_time += pd.Timedelta(minutes=1)

        df = data_df.copy()
        new_df = pd.DataFrame()
        for tic in ticker_list:
            tmp_df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], index=times
            )
            tic_df = df[df.tic == tic]
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i]["timestamp"]] = tic_df.iloc[i][
                    ["open", "high", "low", "close", "volume"]
                ]

                if str(tmp_df.iloc[0]["close"]) == "nan":
                    for i in range(tmp_df.shape[0]):
                        if str(tmp_df.iloc[i]["close"]) != "nan":
                            first_valid_close = tmp_df.iloc[i]["close"]
                            tmp_df.iloc[0] = [
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                0.0,
                            ]
                            break
                if str(tmp_df.iloc[0]["close"]) == "nan":
                    print(
                        "Missing data for ticker: ",
                        tic,
                        " . The prices are all NaN. Fill with 0.",
                    )
                    tmp_df.iloc[0] = [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]

            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]["close"]) == "nan":
                    previous_close = tmp_df.iloc[i - 1]["close"]
                    if str(previous_close) == "nan":
                        previous_close = 0.0
                    tmp_df.iloc[i] = [
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        0.0,
                    ]
            tmp_df = tmp_df.astype(float)
            tmp_df["tic"] = tic
            new_df = pd.concat([new_df, tmp_df])

        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "timestamp"})

        df["VIXY"] = 0
        turb_df = self.api.get_bars(["VIXY"], time_interval, limit=1).df
        latest_turb = turb_df["close"].values

        df = self.add_technical_indicators(new_df)

        df = self.preprocess_data(df)

        price_array, tech_array, turbulence_array = self.df_to_array(
            df, tech_indicator_list, if_vix=True
        )
        latest_price = price_array[-1]
        latest_tech = tech_array[-1]

        return latest_price, latest_tech, latest_turb