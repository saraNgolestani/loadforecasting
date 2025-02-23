import pandas as pd
from typing import Optional, Tuple
from data_processing import DataLoader
from models import MODELS
from sklearn.metrics import mean_absolute_percentage_error

class Diagnose_Forecast():

    def __init__(self, validation_period: int, models: list, winter: bool, train_start : pd.Timestamp, train_end : pd.Timestamp):
        self.train_start = train_start
        self.train_end = train_end
        self.validation_period = validation_period
        self.models = models
        self.winter = winter
        self.dataloader = DataLoader(history_start=self.train_start, history_end=self.train_end+pd.Timedelta(self.validation_period), winter=self.winter)
        self.season_df = self.dataloader.get_seasonal_vals()
        self.season_df.set_index('DATETIME', inplace=True)

    def prepare_test_train(self):

        if self.train_start and self.train_end:
            train_data = self.season_df[(self.season_df.index >= self.train_start) & (self.season_df.index <= self.train_end)]
        else:
            train_data = self.season_df[self.season_df.index < (self.season_df.index.max() - pd.Timedelta(days=self.validation_period))]
        valid_data = self.season_df[self.season_df.index >= (self.season_df.index.max() - pd.Timedelta(days=self.validation_period))]
        self.features = list(self.season_df.columns.values)
        self.features.remove('AIL_ACTUAL')
        self.X_train = train_data[self.features]
        self.Y_train = train_data['AIL_ACTUAL']
        self.X_valid = valid_data[self.features]
        self.Y_valid = valid_data['AIL_ACTUAL']
        return
    

    def diagnose(self) -> pd.DataFrame:
        
        models_metrics = pd.DataFrame()
        self.prepare_test_train()
        models = MODELS(x_train=self.X_train, y_train=self.Y_train, x_test=self.X_valid, y_test=self.Y_valid)
        for model_n in self.models:
            best_model, best_params, feature_importance_df, performance = models.train(model_n)
            model_result = {
            'Model': best_model,
            'Model_Name': model_n,
            'Best_Params': best_params,
            'Performance': performance,
            'Feature_Importance': feature_importance_df
        }

            models_metrics = pd.concat([models_metrics, pd.DataFrame([model_result])], ignore_index=True)

        return models_metrics


    def safe_replace_year(self, dt, new_year):
        try:
            return dt.replace(year=new_year)
        except ValueError:
            # Handle February 29 in non-leap years by moving to February 28
            if dt.month == 2 and dt.day == 29:
                return dt.replace(year=new_year, day=28)
            else:
                raise


    def forecast(self, forecast_start: pd.Timestamp, forecast_end: pd.Timestamp, wthr_year:int, featured_metric: str) -> pd.DataFrame:

        master_df = self.dataloader.get_master_df()
        master_df.set_index('DATETIME', inplace=True)
        df = master_df[master_df.index.year == wthr_year]
        X_pred = pd.DataFrame()
        forecast_end = forecast_end + pd.Timedelta(days=1)
        #the following should be for all wthr related or scenario related values.
        wthr_df = df[['TEMP_LAG_2H', 'TEMPERATURE', 'EXTREME_COLD']]
        wthr_df.index = wthr_df.index.map(lambda dt: self.safe_replace_year(dt, forecast_start.year))
        wthr_df = wthr_df[(wthr_df.index >= forecast_start) & (wthr_df.index <= forecast_end)]
        date_range = pd.date_range(forecast_start, forecast_end, freq='h')
        # prepare the calendar related values
        X_pred['DATETIME'] = date_range
        X_pred['SUN_UP'] = X_pred['DATETIME'].dt.hour.apply(lambda x: 1 if 7 <= x <= 10 or 16 <= x <= 21 else 0)
        X_pred['MONTH'] = X_pred['DATETIME'].dt.month
        X_pred['DAY'] = X_pred['DATETIME'].dt.day
        X_pred['HE'] = X_pred['DATETIME'].dt.hour + 1
        X_pred['DAYOFWEEK'] = X_pred['DATETIME'].dt.dayofweek
        X_pred['WEEKEND'] = X_pred['DAYOFWEEK'].apply(lambda x: 1 if x >= 5 else 0)
        X_pred.set_index('DATETIME', inplace=True)
        X_pred[['TEMP_LAG_2H', 'TEMPERATURE', 'EXTREME_COLD']]= wthr_df[['TEMP_LAG_2H', 'TEMPERATURE', 'EXTREME_COLD']]
        
        Y_pred_true = master_df[(master_df.index >= forecast_start) & (master_df.index <= forecast_end)]['AIL_ACTUAL']

        models_metrics = self.diagnose()
        X_pred = X_pred[self.features]
        best_mape_index = models_metrics['Performance'].apply(lambda x: x[featured_metric]).idxmin()
        model = models_metrics['Model'][best_mape_index]
        print(models_metrics['Model_Name'][best_mape_index])
        Y_pred = model.predict(X_pred)
        if len(Y_pred_true) == len(Y_pred):
            mape  = mean_absolute_percentage_error(Y_pred_true, Y_pred)
        else: mape = 'NA'
        return master_df, X_pred, Y_pred, Y_pred_true, mape

# test = Diagnose_Forecast(train_start=pd.Timestamp('2024-01-01 00:00:00'), train_end= pd.Timestamp('2025-01-31 23:59:59'), validation_period=14, models=['XGBoost', 'Randomforest', 'LinearRegression'], winter=True)
# _, _, _, _, mape = test.forecast(forecast_start=pd.Timestamp('2025-01-15 00:00:00'), forecast_end=pd.Timestamp('2025-01-31 00:00:00'), wthr_year=2025, featured_metric='daily_peak_mape')
# print(mape)

