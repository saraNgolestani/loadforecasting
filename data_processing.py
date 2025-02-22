
import pandas as pd
path = r'C:\Users\snaserigolestani\Downloads\loadforecasting\data\Load and Temp Hist Data.csv'

class DataLoader():
     def __init__(self, path: str, history_start:pd.Timestamp, history_end:pd.Timestamp):
        self.path = path
        self.history_start = history_start
        self.history_end = history_end
        self.master_df = None
        self.date_range = pd.date_range(history_start, history_end, freq='h')

     def get_master_df(self)->pd.DataFrame:

        df = pd.read_csv(path)
        # Setting an standard datetime index 
        df['ADJ_DATE'] = pd.to_datetime(df['DEL_DATE']) + pd.to_timedelta((df['HE'] == 24).astype(int), unit='d')
        df['HB'] = df['HE'].replace(24, 0)
        df['DATETIME'] = pd.to_datetime(df['ADJ_DATE'].dt.strftime('%Y-%m-%d') + ' ' + df['HB'].astype(str) + ':00')
        df.drop(columns=['ADJ_DATE', 'DEL_DATE', 'HB'], inplace=True)
        self.master_df = df
        return df 


     def get_seasonal_vals(self) -> pd.DataFrame:
        self.get_master_df()
        winter_m = [11, 12, 1, 2, 3, 4]
        summer_m = [5, 6, 7, 8, 9, 10]

        self.winter_load = self.master_df[self.master_df['DATETIME'].dt.month.isin(winter_m)]['DATETIME', 'AIL_ACTUAL']
        self.winter_wthr = self.master_df[self.master_df['DATETIME'].dt.month.isin(winter_m)]['DATETIME', 'HE', 'TEMPERATURE']

        self.summer_load = self.master_df[self.master_df['DATETIME'].dt.month.isin(summer_m)]['DATETIME', 'AIL_ACTUAL']
        self.summer_wthr = self.master_df[self.master_df['DATETIME'].dt.month.isin(summer_m)]['DATETIME', 'HE', 'TEMPERATURE']

        return