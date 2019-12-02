import plotly.graph_objects as go

import pandas as pd
df = pd.read_csv('mortgate_data_predictions_new.csv')

for col in df.columns:
    df[col] = df[col].astype(str)

df['text'] = 'State: ' + df['State'] + '<br>' + \
    'Delinquent: ' + df['Delinquent'] + '<br>' + \
    'Not Delinquent: ' + df['Not Delinquent']

fig = go.Figure(data=go.Choropleth(
    locations=df['State Abr'],
    z=df['% Predicted'].astype(float),
    locationmode='USA-states',
    colorscale='Blues',
    autocolorscale=False,
    text=df['text'],  # hover text
    marker_line_color='black',  # line markers between states
    colorbar_title="Delinquency (%)"
))

fig.update_layout(
    title_text='Mortgage Loan Delinquency Predictions',
    geo=dict(
        scope='usa',
        projection=go.layout.geo.Projection(type='albers usa'),
        showlakes=True,  # lakes
        lakecolor='rgb(255, 255, 255)'),
)

fig.show()
