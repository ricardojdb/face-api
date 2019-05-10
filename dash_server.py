#!/usr/bin/env python
# coding: utf-8

from dash.dependencies import Output, Input
from datetime import datetime, timedelta
from collections import deque

import mysql.connector
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import plotly
import random
import base64
import dash
import time

#host_="localhost"
host_= "localhost"
user_="admin"
passwd_="admin"
database_="facedb"

colors = [(150, 150, 150), (8, 154, 255),(0, 253, 255),(231, 217, 0),(0, 0, 190),(0, 184, 113), (98, 24, 91)]
colors = [tuple(reversed(i)) for i in colors]
classes = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

#image_filename = 'src/everis.png' # replace with your own image
#encoded_image = base64.b64encode(open(image_filename, 'rb').read())

app = dash.Dash(__name__)
app.layout = html.Div([html.Div([dcc.Graph(id='live-pie-chart',style={'display':'inline-block', 'width': '100%',}),
                                dcc.Interval(id='pie-chart-update', 
                                             interval=1*1000,  n_intervals=0)],
                                style={'float': 'left', 'width': '33%','text-align': 'right'},),
                       
                       html.Div([dcc.Graph(id='live-bar-chart', style={'display':'inline-block', 'width': '100%'}),
                                 dcc.Interval(id='bar-chart-update', 
                                              interval=1*1000, n_intervals=0)],
                                style={'float': 'left', 'width': '33%', 'text-align': 'middle'},),
                       
                       html.Div([dcc.Graph(id='live-emotions-bar', style={'display':'inline-block', 'width': '100%'}),
                                 dcc.Interval(id='emotions-update', 
                                              interval=1*1000, n_intervals=0)],
                                style={'float': 'left', 'width': '33%', 'text-align': 'left'},),
                    
                       html.Div([dcc.Graph(id='live-graph', style={'width': '75%', 'display': 'inline-block'} ),
                                 dcc.Interval(id='graph-update',
                                              interval=1*1000, n_intervals=0)],
                                style={'float': 'left', 'width': '100%', 'text-align': 'center'},),                     
                      ],)


@app.callback(Output('live-graph', 'figure'),
              [Input('graph-update', 'interval')])
def update_graph_scatter(n):
    try:
        #conn = sqlite3.connect('face_sentiment.db', timeout=10)
        conn = mysql.connector.connect(host=host_, user=user_,
                                      passwd=passwd_, database=database_) 
        cursor = conn.cursor()
        df = pd.read_sql("SELECT * FROM sentiment ORDER BY time DESC LIMIT 500", conn)
        df.sort_values('time', inplace=True)

        df['date'] = pd.to_datetime(df['time'])
        df.set_index('date', inplace=True)

        df = df.resample('1s').mean()
        df.dropna(inplace=True)
        Y = []
        X = df.index[-100:]
        for label in classes:
            Y.append(df[label].values[-100:])

        data_list = []
        for i in range(7):
            data = plotly.graph_objs.Scatter(x=X, y=Y[i], showlegend=True,
                                             name=classes[i], mode='lines+markers',
                                             line=dict(color='rgb({},{},{})'.format(*colors[i])))
            data_list.append(data)
                    
        time_now = datetime.now()
        time_max = pd.to_datetime(time_now)
        time_min =  pd.to_datetime(time_now-timedelta(seconds=20))   
        
        conn.close()
        return {'data': data_list,'layout' : go.Layout(xaxis=dict(range=[time_min, time_max]),
                                                       yaxis=dict(range=[0,1]),
                                                       title='Sentiment')}

    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))

@app.callback(Output('live-pie-chart', 'figure'),
              [Input('pie-chart-update', 'interval')])
def update_graph_pie(n):
    try:
        
        #conn = sqlite3.connect('face_sentiment.db', timeout=10)
        conn = mysql.connector.connect(host=host_, user=user_,
                                      passwd=passwd_, database=database_)
        cursor = conn.cursor()
        df = pd.read_sql("SELECT gender FROM sentiment ORDER BY time DESC LIMIT 1000", conn)
        
        df_male = df.loc[df['gender'] == 'Male']
        male_num = len(df_male)
        df_female = df.loc[df['gender'] == 'Female']
        female_num = len(df_female)

        Labels = ['Masculino','Femenino']
        values = [male_num,female_num]

        common_props = dict(labels=Labels, values=values)

        data = plotly.graph_objs.Pie(
                **common_props,
                textinfo='percent',
                textposition='outside',
                hole = .4,
                textfont=dict(size=12),
                marker=dict(line=dict(color='#FFFFFF', width=1)),
                sort = False)
        
        conn.close()
        return {'data': [data],'layout' : go.Layout(title='Sexo')} 

    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')

@app.callback(Output('live-bar-chart', 'figure'),
              [Input('bar-chart-update', 'interval')])
def update_graph_bar(n):
    try:
        #conn = sqlite3.connect('face_sentiment.db', timeout=10)
        conn = mysql.connector.connect(host=host_, user=user_,
                                      passwd=passwd_, database=database_)

        df = pd.read_sql("SELECT age FROM sentiment ORDER BY time DESC LIMIT 1000", conn)
        
        age_range = [[0,18],[19,25],[26,30],[31,39],[40,49],[50,200]]

        values = np.array([sum(df.age.between(age[0], age[1])) for age in age_range])
        
        Labels = ['0-18','19-25','26-30','31-39','40-49','50-mayor']
        
        total = np.sum(values) + 1e-6
        values_percentage = np.around((values / total)*100)

        data = plotly.graph_objs.Bar(
                x = Labels,
                y = values_percentage,
                text=[str(x)+'%' for x in values_percentage],
                textposition = 'outside')

        conn.close()
        return {'data': [data],'layout' : go.Layout(title='Edad',
                                                    yaxis = dict(title= 'Porcentaje (%)', range = [1, 105]),
                                                    xaxis = dict(title= 'Rango de edades'))}

    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')


@app.callback(Output('live-emotions-bar', 'figure'),
              [Input('emotions-update', 'interval')])
def update_graph_main_emotions(n):
    try:
        #conn = sqlite3.connect('face_sentiment.db', timeout=10
        conn = mysql.connector.connect(host=host_, user=user_,
                                      passwd=passwd_, database=database_)
        cursor = conn.cursor()
        
        df = pd.read_sql("SELECT neutral, happiness, surprise, sadness, anger, disgust, fear, time FROM sentiment ORDER BY time DESC LIMIT 1000", conn)
        
        emotion_mean = df.mean()
        emotion_mean = round(emotion_mean*100)
        
        Labels = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

        data = plotly.graph_objs.Bar(
                x = Labels,
                y = emotion_mean,
                text=[str(int(x))+'%' for x in emotion_mean],
                textposition = 'outside'
                )

        conn.close()
            
        return {'data': [data],'layout' : go.Layout(title='Resumen de emociones',
                                                    yaxis = dict(title= 'Porcentaje Promedio (%)', range = [1, 105]),
                                                    xaxis = dict(title= 'Emociones'))}

    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')

# app.scripts.config.serve_locally = True
# app.css.config.serve_locally = True
app.run_server(host='0.0.0.0')
