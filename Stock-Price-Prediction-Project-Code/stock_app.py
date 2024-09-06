import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = dash.Dash()
server = app.server

scaler = MinMaxScaler(feature_range=(0,1))

# Load cryptocurrency datasets
df_btc = pd.read_csv("./BTC-USD.csv")
df_eth = pd.read_csv("./ETH-USD.csv")
df_ada = pd.read_csv("./ADA-USD.csv")

# Function to process each dataset similarly
def process_data(df):
    df["Date"] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
    df.index = df['Date']
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
    for i in range(0, len(data)):
        new_data["Date"][i] = data['Date'][i]
        new_data["Close"][i] = data["Close"][i]
    new_data.index = new_data.Date
    new_data.drop("Date", axis=1, inplace=True)
    return new_data

# Process BTC-USD data
btc_data = process_data(df_btc)

# Split the data into training and validation
dataset = btc_data.values
train = dataset[0:987, :]
valid = dataset[987:, :]

# Scaling data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60, len(train)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Load LSTM model
model = load_model("ADA_USD.h5")
#model = load_model("BTC_USD.h5")
#model = load_model("ETH_USD.h5")

# Prepare test data for prediction
inputs = btc_data[len(btc_data)-len(valid)-60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make predictions
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

# Prepare validation data for visualization
train = btc_data[:987]
valid = btc_data[987:]
valid['Predictions'] = closing_price

# Define Dash layout and callbacks
app.layout = html.Div([
    html.H1("Cryptocurrency Price Analysis Dashboard", style={"textAlign": "center"}),

    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Data', children=[
            html.Div([
                html.H2("Actual closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data",
                    figure={
                        "data": [
                            go.Scatter(
                                x=train.index,
                                y=valid["Close"],
                                mode='markers'
                            )
                        ],
                        "layout": go.Layout(
                            title='Scatter plot',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                ),
                html.H2("LSTM Predicted closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data",
                    figure={
                        "data": [
                            go.Scatter(
                                x=valid.index,
                                y=valid["Predictions"],
                                mode='markers'
                            )
                        ],
                        "layout": go.Layout(
                            title='Scatter plot',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                )
            ])
        ]),

        dcc.Tab(label='Other Cryptocurrencies', children=[
            html.Div([
                html.H1("Cryptocurrency High vs Lows", style={'textAlign': 'center'}),

                dcc.Dropdown(id='crypto-dropdown',
                             options=[{'label': 'Bitcoin', 'value': 'BTC'},
                                      {'label': 'Ethereum', 'value': 'ETH'},
                                      {'label': 'Cardano', 'value': 'ADA'}],
                             multi=True, value=['BTC'],
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),

                html.H1("Cryptocurrency Market Volume", style={'textAlign': 'center'}),

                dcc.Dropdown(id='volume-dropdown',
                             options=[{'label': 'Bitcoin', 'value': 'BTC'},
                                      {'label': 'Ethereum', 'value': 'ETH'},
                                      {'label': 'Cardano', 'value': 'ADA'}],
                             multi=True, value=['BTC'],
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
            ])
        ])
    ])
])

@app.callback(Output('highlow', 'figure'),
              [Input('crypto-dropdown', 'value')])
def update_highlow_graph(selected_dropdown):
    dropdown = {"BTC": "Bitcoin", "ETH": "Ethereum", "ADA": "Cardano"}
    trace1 = []
    trace2 = []
    for crypto in selected_dropdown:
        df = pd.read_csv(f"./{crypto}-USD.csv")
        trace1.append(
            go.Scatter(x=df["Date"], y=df["High"], mode='lines', opacity=0.7,
                       name=f'High {dropdown[crypto]}', textposition='bottom center'))
        trace2.append(
            go.Scatter(x=df["Date"], y=df["Low"], mode='lines', opacity=0.6,
                       name=f'Low {dropdown[crypto]}', textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(
                  title=f"High and Low Prices for {', '.join(dropdown[i] for i in selected_dropdown)} Over Time",
                  xaxis={'title': 'Date', 'rangeslider': {'visible': True}},
                  yaxis={'title': 'Price (USD)'}
              )}
    return figure

@app.callback(Output('volume', 'figure'),
              [Input('volume-dropdown', 'value')])
def update_volume_graph(selected_dropdown):
    dropdown = {"BTC": "Bitcoin", "ETH": "Ethereum", "ADA": "Cardano"}
    trace1 = []
    for crypto in selected_dropdown:
        df = pd.read_csv(f"./{crypto}-USD.csv")
        trace1.append(
            go.Scatter(x=df["Date"], y=df["Volume"], mode='lines', opacity=0.7,
                       name=f'Volume {dropdown[crypto]}', textposition='bottom center'))
    data = trace1
    figure = {'data': data,
              'layout': go.Layout(
                  title=f"Market Volume for {', '.join(dropdown[i] for i in selected_dropdown)} Over Time",
                  xaxis={'title': 'Date', 'rangeslider': {'visible': True}},
                  yaxis={'title': 'Transactions Volume'}
              )}
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)
