from flask import Flask, render_template, request, flash
from data import Articles
import iex
import datetime as date
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

@app.route("/model", methods=['GET'])
def model():
	return render_template('model.html')

# @app.route("/stock/<ticker>")
# def chartTicker(ticker):
# 	today = date.datetime.now().strftime('%Y%m%d')
# 	df = iex.stockMinData(ticker, today, today)
# 	return render_template('stock.html', ticker=ticker.upper(), df= df.to_html(classes=["table-bordered", "table-striped", "table-hover"]))

@app.route("/stock", methods=['GET', 'POST'])
def stock():
	ticker = ''
	today = date.datetime.now().strftime('%Y%m%d')
	if request.method == 'POST':
		ticker = request.form['stockSearch']
		df = iex.stockMinData(ticker, today, today)
		return render_template('stock.html', ticker=ticker.upper(), df= df.to_html(classes=["table-bordered", "table-striped", "table-hover"]))
	else:
		return render_template('stock.html')

if __name__ == "__main__":
    app.run(debug=True)