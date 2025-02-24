from dash import dcc, html, Input, Output, State
import dash
import plotly.graph_objects as go
import pandas as pd
from diagnose_forecast import Diagnose_Forecast

app = dash.Dash(__name__)


#Set the layout for inputs and outputs we can always extened the manual options but for the purpose of simplicity, the selection of ranges and inputs are limited here. 
app.layout = html.Div([html.H1('Historical and Forecasting Dashboard'),
                       html.Label("Select Forecasting Period:"),
                       dcc.DatePickerRange(
                           id='forecast-period',
                           start_date=pd.Timestamp('2024-10-20 00:00:00'),
                           end_date=pd.Timestamp('2024-10-30 23:59:59'),
                           display_format='YYYY-MM-DD HH'

                       ), 
                        html.Br(),
                        html.Label("Select Season"),
                        dcc.Dropdown(
                            id='season',
                            options=[
                                {'label': 'Winter', 'value': True},
                                {'label': 'Summer', 'value': False}
                            ],
                            value=False
                        ),
                       html.Br(),
                       html.Label('Select Weather Year:'), 
                       dcc.Dropdown(id='wthr-year', 
                                    options=[{'label': str(year), 'value': year} for year in range(2021, 2026)]
                                    , value=2023),
                       html.Br(),
                       html.Label('Select your Featured Metric:'), 
                       dcc.Dropdown(id = 'featured-metric', 
                                    options=[{'label': 'MAPE', 'value': 'mape'},
                                             {'label':'MAE', 'value': 'mae'},
                                             {'label': 'DAILY_MAPE', 'value':'daily_peak_mape'}],
                                             value='mape'),
                        
                        html.Button('Run Pipeline', id='run_forecast', n_clicks=0),
                        html.Div(id='mape_output'),
                        dcc.Graph(id='AIL')])


# define the callback variables
@app.callback(
[Output('AIL', 'figure'),
 Output('mape_output', 'children')],
 [Input('run_forecast', 'n_clicks')],
 [State('forecast-period', 'start_date'),
     State('forecast-period', 'end_date'),
     State('season', 'value'),
     State('wthr-year', 'value'),
     State('featured-metric', 'value')]
 )

def update_forecast(n_clicks, start_date, end_date, season, wthr_year, featured_metric):
    if n_clicks == 0:
        return go.Figure(), ''
    
    #the following is the core of the training process, with the training and validation range and also defining which models to run this process with. 
    diagfcst_object = Diagnose_Forecast(validation_period=14, 
                                        models=['XGBoost', 'Randomforest', 'LinearRegression'],
                                        winter=season,
                                        train_start=pd.Timestamp('2024-01-01 00:00:00'), train_end=pd.Timestamp('2025-01-31 23:59:59'))
    

    #After diagnose, the forecasting modelu would choose the best featured parameter between the given models and runs the forecast with that. 
    Historical_df, X_pred, Y_pred, Y_pred_true, mape = diagfcst_object.forecast(
        forecast_start=pd.Timestamp(start_date),
        forecast_end=pd.Timestamp(end_date),
        wthr_year=wthr_year,
        featured_metric=featured_metric,
    )

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=Historical_df.index, y=Historical_df['AIL_ACTUAL'], mode='lines', name='Historical'))
    figure.add_trace(go.Scatter(x=X_pred.index, y=Y_pred, mode='lines', name='Forecast'))
    figure.update_layout(title='Historical and Forecast values', xaxis_title='Date', yaxis_title='MW')
    mape_text = f"MAPE: {round(mape, ndigits=4)} for the period of {start_date} to {end_date}" if mape != 'NA' else f"MAPE: Not Available for the period of {start_date} to {end_date}"

    return figure, mape_text

if __name__ == '__main__':
    app.run_server(debug = True)