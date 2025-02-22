import pandas as pd
from typing import Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

class MODELS:

    def __init__(self, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: Optional[pd.DataFrame] = None, y_test: Optional[pd.DataFrame] = None):

        self.x_train = x_train
        self.y_train = y_train
        self.y_test = y_test
        self.x_test = x_test
        self.features = list(self.x_test.columns.values)
        self.models_info = {}

    
    def train(self, model_name):
        if model_name == 'Randomforest':
            param_grid = {
                'n_estimators': [10, 100],
                'max_depth': [5, 15],
                'min_samples_leaf' :[6, 8, 10]
            }
            grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1,scoring='neg_mean_absolute_percentage_error')
        elif model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [10, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [4, 6]
            }
            grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
       
        grid_search.fit(self.x_train, self.y_train)
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        feature_importance_df = pd.DataFrame({'Feature': self.features, 'Importance': best_model.feature_importances_}).sort_values(by="Importance",ascending=False).reset_index(drop=True)
        performance, self.y_pred = self.predict(best_model)

        return best_model, best_params, feature_importance_df, performance

    
    def daily_peak_mape(self):
        y_test_max_by_date = self.y_test.groupby(self.y_test.index.day).max()

        y_pred_max_by_date = [max(self.y_pred[self.y_test.index.day == date]) for date in y_test_max_by_date.index]

        return mean_absolute_percentage_error(y_test_max_by_date, y_pred_max_by_date)

    def predict(self, best_model) -> Tuple:

        self.y_pred = best_model.predict(self.x_test)

        for i, v in enumerate(self.y_pred):
            if v < 0:
                self.y_pred[i] = 0

        if self.y_test is not None: 
            performance = {
                'mae': mean_absolute_error(self.y_test, self.y_pred),
                'mape': mean_absolute_percentage_error(self.y_test, self.y_pred),
                'daily_peak_mape': self.daily_peak_mape()
            }
        
        return performance, self.y_pred


    

