# Dataset was taken from Kaggle: https://www.kaggle.com/tsarkov90/crime-in-russia-20032020

# libraries
import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import datetime as dt   # for converting some date
import statsmodels.api as sm    # needed for correlation line in corr_plot



df = pd.read_csv('https://raw.githubusercontent.com/Lotaristo/Datasets1/main/Crime')

df.month = pd.to_datetime(df.month)     # converted to datetime
df['year'] = df.month.dt.year   # extracted year
df.month = df['month'].apply(lambda x: dt.datetime.strftime(x, '%Y-%d-%m')) # changed day to the month, because format isn't correct
year = df.pop('year')   # replaced created column from end to the the beginnig of df
df.insert(0, 'year', year)

df2 = round(df.groupby(year).mean())    # grouped by years and rounded

df3 = df.iloc[:, 2:]    # for some reason I didn't managed to use .iloc without errors, hence I just created new ds without dates columns





# layout for app:
#__title
#__dropdown1__, _-dropdown2__
#__graph1__, __graph2__
#__dropdown3__, __dropdown4__, __dropdown4__
#__graph3__, __graph4__

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )
server = app.server

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            html.H1('Crimes in Russia', className = 'text-center text-primary')

        )
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(id='average',
                         multi=True,    # enabled for multiple selections
                         value='Total_crimes',  # standard valuer
                         options=[{'label':i, 'value':i} for i in df3.columns.unique()] # choosed unique columns
                         )
        ], width = 5),  # width 5 of 12
        dbc.Col([
            dcc.Dropdown(id='timelapse',
                         multi=False,
                         value='Total_crimes',
                         options=[{'label':i, 'value':i} for i in df3.columns.unique()]
                         )
        ], width = {'size':5, 'offset':1})  # size is same as width, offset to move a little bit to the right
    ], justify='start'),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='average_graph', figure={}) # created graphs
        ], width = 5),
        dbc.Col([
            dcc.Graph(id='timelapse_graph', figure={})
        ], width=7),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(id='heatmap', # honestly, there is no need in this, but I didn't manage to do it without Input
                         multi=False,
                         value='Total_crimes',
                         disabled=True, # locked because it's used like a title, not a dropdown menu
                         options=[{'label': i, 'value': i} for i in df3.columns.unique()]
                         )
        ], width=4),
        dbc.Col([
            dcc.Dropdown(id='corr1', # 2 graphs to choose one value for corr_graph
                         multi=False,
                         value='Total_crimes',
                         options=[{'label': i, 'value': i} for i in df3.columns.unique()]
                         )
        ], width={'size':3, 'offset':2}),
        dbc.Col([
            dcc.Dropdown(id='corr2',
                         multi=False,
                         value='Huge_damage',
                         options=[{'label': i, 'value': i} for i in df3.columns.unique()]
                         )
        ], width=3)
        ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='heatmap_graph', figure={})
        ], width = 5),
        dbc.Col([
            dcc.Graph(id='corr_graph', figure={})
        ], width = 7)
    ])
])

@app.callback(
    Output('average_graph', 'figure'),
    Input('average', 'value')
)
def update_average_graph(val):
    a_fig = px.line(df2,
                   x = 'year',
                   y = val)
    a_fig.update_layout(title = # this section used to align title to the center
                        {'text': "Average Amount of Crimes per Year",
                         'y':0.9, 'x':0.5,
                         'xanchor': 'center', 'yanchor': 'middle'})
    return a_fig


@app.callback(
    Output('timelapse_graph', 'figure'),
    Input('timelapse', 'value')
)
def update_timelapse_graph(val):
    t_fig = px.bar(df,
                   x = 'month',
                   y = val,
                   color='year',
                   color_continuous_scale=px.colors.cyclical.HSV) # changed to make year columns more distinct from each other
    t_fig.update_layout(title=
                        {'text': "Total Amount of Selected Crime by Month for All Time",
                         'y': 0.9, 'x': 0.5,
                         'xanchor': 'center', 'yanchor': 'middle'})

    return t_fig


@app.callback(
    Output('heatmap_graph', 'figure'),
    Input('heatmap', 'value') # there is no need in Input because there is no values to be changed, but I didn't manage to do graph withoit it
)
def heatmap_graph(val):
    h_fig = px.imshow(df2.iloc[:, 3:],
                      color_continuous_scale='matter')
    h_fig.update_yaxes(autorange=True) # y-axis was inverted, this command will return it's to original position
    h_fig.update_layout(title=
                        {'text': "Average Amount of Various Crimes per Year on a Heatmap",
                         'y': 0.9, 'x': 0.5,
                         'xanchor': 'center', 'yanchor': 'middle'})
    return h_fig


@app.callback(
    Output('corr_graph', 'figure'),
    Input('corr1', 'value'),
    Input('corr2', 'value'), # takes two inputs and give it to func
)
def update_scatter_graph(val1, val2):
    s_fig = px.scatter(df,
                       x=val1,
                       y=val2,
                       color='year',
                       color_continuous_scale='Mint', # distinction by year
                       template='plotly_white',
                       trendline='lowess') # to better show correlation, need statsmodels
    s_fig.update_layout(title=
                        {'text': "Correlation between Chosen Crimes with Color Distribution by Year",
                         'y': 0.9, 'x': 0.5,
                         'xanchor': 'center', 'yanchor': 'middle'})
    return s_fig

if __name__ == '__main__':
    app.run_server(debug=True) # Use "False" if you run in Jupyter.