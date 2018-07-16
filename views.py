from flask import Flask, render_template, request, flash
from data import Articles
import iex
import ml
import datetime as date
import numpy as np

app = Flask(__name__)

Articles = Articles()
@app.route("/")
def index():
    return render_template('index.html', methods=['GET'])
@app.route("/about")
def about():
	return render_template('about.html', methods=['GET'])
@app.route("/articles")
def articles():
	return render_template('articles.html', articles = Articles)

@app.route("/model", methods=['GET', 'POST'])
def model():
	if request.method == 'POST':
		ticker = request.form['ticker']
		fromDate = date.datetime.strptime(request.form['fromDate'], '%Y-%m-%d')
		fromDate = date.datetime.strftime(fromDate, '%Y%m%d')

		toDate = date.datetime.strptime(request.form['toDate'], '%Y-%m-%d')
		toDate = date.datetime.strftime(toDate, '%Y%m%d')

		interval = request.form['interval']
		indicators = request.form['indicators']
		model = request.form['model']

		if interval == 'daily':
			features, close = ml.dailyRoutine(ticker, fromDate, toDate, indicators)
			valresults, results, neighbors, accScore, precScore, confMatrix = ml.runModel(features, close)
			return render_template('model.html', valresults='Validation Score: {}'.format(valresults), 
				results='Score: {}'.format(results), ticker=ticker, 
				neighbors='Optimal Neighbors: {}'.format(neighbors), 
				accScore='Accuracy Score: {}'.format(accScore),
				precScore='Precision Score: {}'.format(precScore), 
				confMatrix='Confusion Matrix: {}'.format(confMatrix))

		elif interval == 'minute':
			features, close = ml.minuteRoutine(ticker, fromDate, toDate, indicators)
			valresults, results, neighbors, accScore, precScore, confMatrix = ml.runModel(features, close)
			return render_template('model.html', valresults='Validation Score: {}'.format(valresults), 
				results='Score: {}'.format(results), ticker=ticker, 
				neighbors='Optimal Neighbors: {}'.format(neighbors), 
				accScore='Accuracy Score: {}'.format(accScore),
				precScore='Precision Score: {}'.format(precScore), 
				confMatrix='Confusion Matrix: {}'.format(confMatrix))
	else:
		return render_template('model.html')

@app.route("/stock", methods=['GET', 'POST'])
def stock():
	if request.method == 'POST':
		ticker = ''
		today = date.datetime.now().strftime('%Y%m%d')
		try:
			ticker = request.form['stockSearch']
			df = iex.stockMinData(ticker, today, today)
		except:
			return render_template('stock.html', ticker= ticker.upper() + ' is not a valid ticker')
		return render_template('stock.html', ticker=ticker.upper(), df= df.to_html(classes=["table-bordered", "table-striped", "table-hover"]))
	else:
		return render_template('stock.html')

if __name__ == "__main__":
    app.run(debug=True)