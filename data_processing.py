
import pandas as pd

class DataLoader():
     def __init__(self,  history_start:pd.Timestamp, history_end:pd.Timestamp, winter: bool):
        self.path =  r'./data/Load and Temp Hist Data.csv'
        self.history_start = history_start
        self.history_end = history_end
        self.master_df = None
        self.date_range = pd.date_range(history_start, history_end, freq='h')
        self.winter = winter

     def get_master_df(self)->pd.DataFrame:

        df = pd.read_csv(self.path)
        # Setting an standard datetime index 
        df['ADJ_DATE'] = pd.to_datetime(df['DEL_DATE']) + pd.to_timedelta((df['HE'] == 24).astype(int), unit='d')
        df['HB'] = df['HE'].replace(24, 0)
        df['DATETIME'] = pd.to_datetime(df['ADJ_DATE'].dt.strftime('%Y-%m-%d') + ' ' + df['HB'].astype(str) + ':00')
        df.drop(columns=['ADJ_DATE', 'DEL_DATE', 'HB'], inplace=True)

        # Extracting calendar features 
        df['SUN_UP'] = df['DATETIME'].dt.hour.apply(lambda x: 1 if 7 <= x <= 10 or 16 <= x <= 21 else 0)
        df['MONTH'] = df['DATETIME'].dt.month
        df['DAY'] = df['DATETIME'].dt.day
        df['DAYOFWEEK'] = df['DATETIME'].dt.dayofweek
        df['WEEKEND'] = df['DAYOFWEEK'].apply(lambda x: 1 if x >= 5 else 0)

        # lagged wthr data for better comprehention
        df['EXTREME_COLD'] = df['TEMPERATURE'].apply(lambda x: 1 if x <= -20 else 0)
        df['TEMP_LAG_2H'] = df['TEMPERATURE'].shift(2).fillna(method='bfill', limit=2)
      #   df = df[df['DATETIME'].isin(self.date_range)]
        self.master_df = df
        return df 


     def get_seasonal_vals(self) -> pd.DataFrame:
        self.get_master_df()
        winter_m = [11, 12, 1, 2, 3, 4]
        summer_m = [5, 6, 7, 8, 9, 10]

        if self.winter:
         self.season_df = self.master_df[self.master_df['DATETIME'].dt.month.isin(winter_m)]
        else: 
         self.season_df = self.master_df[self.master_df['DATETIME'].dt.month.isin(summer_m)]

        return self.season_df