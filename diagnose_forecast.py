import pandas as pd
from typing import Optional, Tuple
from data_processing import DataLoader
from models import MODELS

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
            'Model_Name': model_n,
            'Best_Params': best_params,
            'Performance': performance,
            'Feature_Importance': feature_importance_df
        }

            models_metrics = pd.concat([models_metrics, pd.DataFrame([model_result])], ignore_index=True)

        return models_metrics


test = Diagnose_Forecast(train_start=pd.Timestamp('2021-01-01 00:00:00'), train_end= pd.Timestamp('2025-01-31 23:59:59'), validation_period=14, models=['XGBoost'], winter=False)
metrics = test.diagnose()
print(metrics['Feature_Importance'][0])
print(metrics['Performance'][0])

