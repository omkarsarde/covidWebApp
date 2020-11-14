from flask import Flask, request, render_template, jsonify,redirect
from Model import test
from math import floor
from bokeh.resources import CDN
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource, FactorRange
from bokeh.embed import components
from datetime import date, timedelta
import urllib.request, pandas, os

app = Flask(__name__)


def download_file(country):
    if country == "UNITED STATES":
        country = 'US'
    name = str(date.today() - timedelta(days=2))
    name = name.split('-')
    name = name[1] + "-" + name[2] + "-" + name[0] + ".csv"
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/' + name
    urllib.request.urlretrieve(url, 'jhu_data.csv')
    df = pandas.read_csv('jhu_data.csv')
    df.drop(df.loc[df['Province_State'] == 'Unknown'].index, inplace=True)
    df.drop(axis=0, columns=['FIPS', 'Admin2', 'Last_Update', 'Lat', 'Long_'], inplace=True)
    df['Country_Region'] = df['Country_Region'].str.upper()
    data = df[df['Country_Region'] == country]
    if country == 'US' or country == 'UNITED KINGDOM':
        data['Province_State'].replace({"Recovered":'z1: Total Recovered Cases'},inplace=True)
    data['Province_State'].fillna(country, inplace=True)
    data = data.groupby(['Province_State'], as_index=False).sum()
    x_val = [i for i in range(data.shape[0])]
    labels = {}
    for i in x_val:
        labels[i] = data.iloc[i, 0]
    Confirmed_Cases = list(data['Confirmed'])
    Deaths = list(data['Deaths'])
    Recovered = list(data['Recovered'])
    Active = list(data['Active'])
    os.remove('jhu_data.csv')
    return Confirmed_Cases, Deaths, Recovered, Active, labels


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    input_val = [x for x in request.form.values()]
    country = input_val[0]
    # uncomment this line to deploy full APP
    # model = input_val[1]
    model = 'xgb'
    prediction = test(country, model)
    prediction = floor(prediction)
    input_val = country.upper()
    script, div = plot(input_val)
    return render_template('prediction.html', pred=f'{prediction}',
                           input_val=f'{input_val}', script=script, div=div, resources=CDN.render())

def plot(Country):
    Confirmed_Cases, Deaths, Recovered, Active, labels = download_file(Country)
    v_range = [i for i in range(len(Deaths))]
    if len(v_range) == 1:
        index = list(labels.values())
        vals = ['Confirmed_Cases', 'Deaths', 'Active_Cases', 'Recovered']
        data = {'State': index,
                'Confirmed_Cases': Confirmed_Cases,
                'Deaths': Deaths,
                'Active_Cases': Active,
                'Recovered': Recovered}
        x = [(i, val) for i in index for val in vals]
        counts = sum(zip(data['Confirmed_Cases'], data['Deaths'], data['Active_Cases'], data['Recovered']), ())
        source = ColumnDataSource(data=dict(x=x, counts=counts))
        p = figure(x_range=FactorRange(*x), plot_height=600, title="SARS COV-19 Cases for: "+str(date.today()), tools="",
                   background_fill_color='lightgrey')

        p.vbar(x='x', top='counts', width=0.9, source=source, fill_color='orange')
        p.add_tools(HoverTool(
            tooltips=[
                ('Cases', '@counts'),
            ],
            mode='vline'
        ))
        p.y_range.start = 0
        p.x_range.range_padding = 0.1
        p.xaxis.major_label_orientation = 1
        p.xgrid.grid_line_color = None
        scripts, div = components(p)
        return scripts, div
    else:
        width = 0.5
        p = figure(plot_width=1000, plot_height=600, title="SARS COV-19 Cases for: "+str(date.today()), background_fill_color='lightgrey')
        p.vbar(x=v_range, width=width, bottom=0, top=Confirmed_Cases, color=('yellow'), legend_label='Confirmed_Cases')
        p.vbar(x=v_range, width=width, bottom=0, top=Deaths, color=('red'), legend_label='Deaths')
        p.vbar(x=v_range, width=width, bottom=0, top=Active, color=('blue'), legend_label='Active_Cases')
        p.vbar(x=v_range, width=width, bottom=0, top=Recovered, color=('green'), legend_label='Recovered')
        p.xaxis.ticker = v_range
        p.xaxis.major_label_overrides = labels
        p.add_tools(HoverTool(
            tooltips=[
                ('Cases', '@top{F}'),
            ],
        ))
        p.legend.location = "top_right"
        p.legend.click_policy = "hide"
        p.y_range.start = 0
        p.x_range.range_padding = 0.1
        p.xaxis.major_label_orientation = 1
        p.xgrid.grid_line_color = None
        scripts, div = components(p)
    return scripts, div


@app.route('/api', methods=['GET'])
def api():
    if 'c' in request.args:
        input_val = request.args['c']
        prediction = test(input_val, 'xgb')
        prediction = floor(prediction)
        input_val = input_val.upper()
        Confirmed_Cases, Deaths, Recovered, Active, labels = download_file(input_val)
        pred = {'Nation': input_val, 'Predicted New Cases': prediction,
                'Confirmed_Cases': Confirmed_Cases, 'Deaths': Deaths, 'Recovered_Cases': Recovered,
                'Active_Cases': Active,
                'Labels': labels,
                "Source DISCLAIMER": "Demonstrative ML API. Data collected from ourworldindata.org and JHU"}
    else:
        return "Error: No Country field provided or Invalid Country. Please specify an proper Country."
    return jsonify(pred)


if __name__ == '__main__':
    app.run(debug=False)
