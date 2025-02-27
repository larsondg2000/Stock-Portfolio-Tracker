from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly
import plotly.graph_objects as go
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-replace-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///portfolio.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


# Define models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    stocks = db.relationship('Stock', backref='owner', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Stock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(10), nullable=False)
    account = db.Column(db.String(50), nullable=False)
    shares = db.Column(db.Float, nullable=False)
    cost_basis = db.Column(db.Float, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Create the database and tables
with app.app_context():
    db.create_all()


# Helper functions
def get_current_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        price = stock.info['currentPrice']
        return price
    except Exception as e:
        print(f"Error getting price for {ticker}: {e}")
        return None


def get_dividend_info(ticker, shares):
    stock = yf.Ticker(ticker)
    info = stock.info
    if 'dividendRate' in info and info['dividendRate'] is not None:
        ex_date = info.get('exDividendDate')
        if ex_date:
            ex_date = datetime.fromtimestamp(ex_date).strftime('%m-%d-%Y')
        else:
            ex_date = 'N/A'
        return {
            'ticker': ticker,
            'shares': shares,
            'dividend_rate': info.get('dividendRate', 0),
            'dividend_yield': info.get('dividendYield', 0) if info.get('dividendYield') else 0,
            'payout_ratio': info.get('payoutRatio', 0) * 100 if info.get('payoutRatio') else 0,
            'ex_dividend_date': ex_date,
            'yearly_dividend_total': shares * info.get('dividendRate', 0)
        }
    return None


# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('portfolio'))
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('portfolio'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('portfolio'))
        else:
            flash('Invalid username or password')

    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('portfolio'))

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()

        if existing_user:
            flash('Username or email already exists')
        else:
            new_user = User(username=username, email=email)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user)
            return redirect(url_for('portfolio'))

    return render_template('signup.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/portfolio')
@login_required
def portfolio():
    stocks = Stock.query.filter_by(user_id=current_user.id).all()

    # Create dataframe from stocks
    if stocks:
        data = []
        for stock in stocks:
            current_price = get_current_price(stock.ticker)
            if current_price:
                total_value = stock.shares * current_price
                total_cost = stock.shares * stock.cost_basis
                gain_loss = total_value - total_cost
                gain_loss_percent = (gain_loss / total_cost) * 100 if total_cost > 0 else 0

                data.append({
                    'id': stock.id,
                    'ticker': stock.ticker,
                    'account': stock.account,
                    'shares': stock.shares,
                    'cost_basis': stock.cost_basis,
                    'current_price': current_price,
                    'total_value': total_value,
                    'gain_loss': gain_loss,
                    'gain_loss_percent': gain_loss_percent
                })

        df = pd.DataFrame(data)

        if not df.empty:
            # Calculate totals
            total_value = df['total_value'].sum()
            total_cost = (df['cost_basis'] * df['shares']).sum()
            total_gain_loss = total_value - total_cost
            total_gain_loss_percent = (total_gain_loss / total_cost) * 100 if total_cost > 0 else 0

            """
            # Debug print to console
            print("Portfolio data:")
            for index, row in df.iterrows():
                pct = (row['total_value'] / total_value) * 100
                print(f"{row['ticker']}: ${row['total_value']:.2f} = {pct:.2f}%")
            """

            # Create pie chart
            pie_data = []
            for index, row in df.iterrows():
                pie_data.append({
                    'ticker': row['ticker'],
                    'percentage': (row['total_value'] / total_value) * 100
                })

            pie_fig = go.Figure(data=[go.Pie(
                labels=[item['ticker'] for item in pie_data],
                values=[item['percentage'] for item in pie_data],
                textinfo='label+percent',
                hoverinfo='label+percent',
                textposition='inside'
            )])

            pie_fig.update_layout(
                title_text="Portfolio Composition",
                showlegend=True
            )

            pie_chart = json.dumps(pie_fig, cls=plotly.utils.PlotlyJSONEncoder)

            # Create gain/loss charts
            df_sorted = df.sort_values('gain_loss', ascending=False)

            # Gain/Loss bar chart
            gain_loss_colors = ['green' if x >= 0 else 'red' for x in df_sorted['gain_loss']]
            gain_loss_chart = go.Figure()
            gain_loss_chart.add_trace(
                go.Bar(
                    x=df_sorted['ticker'].tolist(),
                    y=df_sorted['gain_loss'].tolist(),
                    marker_color=gain_loss_colors,
                    text=df_sorted['gain_loss'].apply(lambda x: f"${x:.2f}").tolist(),
                    textposition='auto'
                )
            )
            gain_loss_chart.update_layout(
                title="Total Gain/Loss by Ticker",
                xaxis_title="Ticker",
                yaxis_title="Gain/Loss ($)",
                height=400,
                margin=dict(l=50, r=50, t=50, b=50)
            )

            # Gain/Loss Percentage bar chart
            percent_colors = ['green' if x >= 0 else 'red' for x in df_sorted['gain_loss_percent']]
            percent_chart = go.Figure()
            percent_chart.add_trace(
                go.Bar(
                    x=df_sorted['ticker'].tolist(),
                    y=df_sorted['gain_loss_percent'].tolist(),
                    marker_color=percent_colors,
                    text=df_sorted['gain_loss_percent'].apply(lambda x: f"{x:.2f}%").tolist(),
                    textposition='auto'
                )
            )
            percent_chart.update_layout(
                title="Gain/Loss Percentage by Ticker",
                xaxis_title="Ticker",
                yaxis_title="Gain/Loss (%)",
                height=400,
                margin=dict(l=50, r=50, t=50, b=50)
            )

            # Convert to JSON
            gain_loss_json = json.dumps(gain_loss_chart, cls=plotly.utils.PlotlyJSONEncoder)
            percent_json = json.dumps(percent_chart, cls=plotly.utils.PlotlyJSONEncoder)

            return render_template('portfolio.html',
                                   stocks=data,
                                   total_value=total_value,
                                   total_gain_loss=total_gain_loss,
                                   total_gain_loss_percent=total_gain_loss_percent,
                                   pie_chart=pie_chart,
                                   gain_loss_chart=gain_loss_json,
                                   percent_chart=percent_json)

    return render_template('portfolio.html', stocks=None)


@app.route('/add_stock', methods=['POST'])
@login_required
def add_stock():
    ticker = request.form.get('ticker').upper()
    account = request.form.get('account')
    shares = float(request.form.get('shares'))
    cost_basis = float(request.form.get('cost_basis'))

    # Validate the ticker exists
    try:
        stock = yf.Ticker(ticker)
    except:
        flash(f"Could not verify ticker {ticker}. Please check and try again.")
        return redirect(url_for('portfolio'))

    new_stock = Stock(
        ticker=ticker,
        account=account,
        shares=shares,
        cost_basis=cost_basis,
        user_id=current_user.id
    )

    db.session.add(new_stock)
    db.session.commit()

    flash(f"Added {ticker} to your portfolio!")
    return redirect(url_for('portfolio'))


@app.route('/update_stock/<int:stock_id>', methods=['POST'])
@login_required
def update_stock(stock_id):
    stock = Stock.query.filter_by(id=stock_id, user_id=current_user.id).first_or_404()

    account = request.form.get('account')
    shares = float(request.form.get('shares'))
    cost_basis = float(request.form.get('cost_basis'))

    if shares <= 0:
        db.session.delete(stock)
        flash(f"Removed {stock.ticker} from your portfolio!")
    else:
        stock.account = account
        stock.shares = shares
        stock.cost_basis = cost_basis
        flash(f"Updated {stock.ticker} in your portfolio!")

    db.session.commit()
    return redirect(url_for('portfolio'))


@app.route('/delete_stock/<int:stock_id>')
@login_required
def delete_stock(stock_id):
    stock = Stock.query.filter_by(id=stock_id, user_id=current_user.id).first_or_404()

    db.session.delete(stock)
    db.session.commit()

    flash(f"Removed {stock.ticker} from your portfolio!")
    return redirect(url_for('portfolio'))


@app.route('/dividends')
@login_required
def dividends():
    stocks = Stock.query.filter_by(user_id=current_user.id).all()

    # Get dividend information for each stock
    dividend_data = []
    for stock in stocks:
        div_info = get_dividend_info(stock.ticker, stock.shares)
        if div_info:
            dividend_data.append(div_info)

    if dividend_data:
        df = pd.DataFrame(dividend_data)

        # Calculate totals
        total_yearly_dividends = df['yearly_dividend_total'].sum()
        average_monthly_dividends = total_yearly_dividends / 12
        average_yield = df['dividend_yield'].mean()

        # Dividend by stock pie chart - direct approach
        pie_data = []
        for index, row in df.iterrows():
            pie_data.append({
                'ticker': row['ticker'],
                'percentage': (row['yearly_dividend_total'] / total_yearly_dividends) * 100
            })

        pie_fig = go.Figure(data=[go.Pie(
            labels=[item['ticker'] for item in pie_data],
            values=[item['percentage'] for item in pie_data],
            textinfo='label+percent',
            hoverinfo='label+percent',
            textposition='inside'
        )])

        pie_fig.update_layout(
            title_text="Dividend Percentage by Stock",
            showlegend=True
        )

        pie_chart = json.dumps(pie_fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Create separate charts for each dividend metric

        # Sort data by dividend total for consistent display
        df_sorted = df.sort_values('yearly_dividend_total', ascending=False)

        # Yearly Dividend Totals
        yearly_div_chart = go.Figure()
        yearly_div_chart.add_trace(
            go.Bar(
                x=df_sorted['ticker'].tolist(),
                y=df_sorted['yearly_dividend_total'].tolist(),
                marker_color='darkblue',
                text=df_sorted['yearly_dividend_total'].apply(lambda x: f"${x:.2f}").tolist(),
                textposition='auto'
            )
        )
        yearly_div_chart.update_layout(
            title="Yearly Dividend Totals",
            xaxis_title="Ticker",
            yaxis_title="Yearly Dividend Total ($)",
            height=350,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        # Dividend Yield chart
        yield_chart = go.Figure()
        yield_chart.add_trace(
            go.Bar(
                x=df_sorted['ticker'].tolist(),
                y=df_sorted['dividend_yield'].tolist(),
                marker_color='blueviolet',
                text=df_sorted['dividend_yield'].apply(lambda x: f"{x:.2f}%").tolist(),
                textposition='auto'
            )
        )
        yield_chart.update_layout(
            title="Dividend Yield",
            xaxis_title="Ticker",
            yaxis_title="Dividend Yield (%)",
            height=350,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        # Payout Ratio chart
        # Color coding: green (<50%), yellow (50-70%), red (>70%)
        payout_colors = ['green' if x < 50 else 'yellow' if 50 <= x < 70 else 'red' for x in df_sorted['payout_ratio']]
        payout_chart = go.Figure()
        payout_chart.add_trace(
            go.Bar(
                x=df_sorted['ticker'].tolist(),
                y=df_sorted['payout_ratio'].tolist(),
                marker_color=payout_colors,
                text=df_sorted['payout_ratio'].apply(lambda x: f"{x:.2f}%").tolist(),
                textposition='auto'
            )
        )
        payout_chart.update_layout(
            title="Payout Ratio",
            xaxis_title="Ticker",
            yaxis_title="Payout Ratio (%)",
            height=350,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        # Convert all charts to JSON
        yearly_div_json = json.dumps(yearly_div_chart, cls=plotly.utils.PlotlyJSONEncoder)
        yield_json = json.dumps(yield_chart, cls=plotly.utils.PlotlyJSONEncoder)
        payout_json = json.dumps(payout_chart, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('dividends.html',
                               dividend_data=dividend_data,
                               total_yearly_dividends=total_yearly_dividends,
                               average_monthly_dividends=average_monthly_dividends,
                               average_yield=average_yield,
                               pie_chart=pie_chart,
                               yearly_div_chart=yearly_div_json,
                               yield_chart=yield_json,
                               payout_chart=payout_json)

    return render_template('dividends.html', dividend_data=None)


@app.route('/analysis')
@login_required
def analysis():
    stocks = Stock.query.filter_by(user_id=current_user.id).all()

    if not stocks:
        return render_template('analysis.html', has_data=False)

    # Prepare data for analysis
    stock_data = []
    for stock in stocks:
        current_price = get_current_price(stock.ticker)
        if current_price:
            stock_data.append({
                'ticker': stock.ticker,
                'shares': stock.shares,
                'cost_basis': stock.cost_basis,
                'current_price': current_price,
                'total_value': stock.shares * current_price
            })

    if not stock_data:
        return render_template('analysis.html', has_data=False)

    risk_df = pd.DataFrame(stock_data)
    total_portfolio_value = risk_df['total_value'].sum()
    risk_df['portfolio_percent'] = (risk_df['total_value'] / total_portfolio_value) * 100
    weights_list = risk_df['portfolio_percent'].tolist()
    stock_list = risk_df['ticker'].tolist()

    # Get historical stock price data for the last 5 years
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')

    data = {}
    for stock in stock_list:
        ticker = yf.Ticker(stock)
        history = ticker.history(start=start_date, end=end_date)
        if not history.empty:
            data[stock] = history['Close']

    if not data:
        return render_template('analysis.html', has_data=False)

    df_prices = pd.DataFrame(data)
    if df_prices.empty:
        return render_template('analysis.html', has_data=False)

    df_prices.index = df_prices.index.tz_localize(None)
    df_prices.bfill(inplace=True)

    # Calculate percent change and covariance matrix from price history
    returns_df = df_prices.pct_change(1).dropna()
    if returns_df.empty:
        return render_template('analysis.html', has_data=False)

    vcv_matrix = returns_df.cov()

    # Convert weights to numpy array and ensure it has the right shape
    weights_np = np.array(weights_list) / 100

    # Calculate variance and standard deviation
    try:
        var_p = np.dot(weights_np, np.dot(vcv_matrix, weights_np))
        sd_p = np.sqrt(var_p)

        # Get annual portfolio and individual stock risks
        sd_p_annual = (sd_p * np.sqrt(250)) / 100
        individual_risks = returns_df.std() * np.sqrt(250)

        # Calculate individual Sharpe ratio
        sharpe_individual = (returns_df.mean() / returns_df.std()) * np.sqrt(250)

        # Get values to calculate portfolio Sharpe Ratio
        shares_dict = {row['ticker']: row['shares'] for row in stock_data}
        prices_df = df_prices.copy()

        # Multiply prices by shares to get total value
        for ticker in prices_df.columns:
            if ticker in shares_dict:
                prices_df[ticker] = prices_df[ticker] * shares_dict[ticker]

        # Add a new column that sums each row for total portfolio value
        prices_df['Total Value'] = prices_df.sum(axis=1)

        # Add new column for daily return
        prices_df["return"] = prices_df['Total Value'].pct_change().fillna(0) * 100

        # Assume risk-free rate is 5-year US Treasury yield (as an example)
        risk_free_rate = 0.03  # 3% annualized

        # Calculate daily portfolio return correctly
        daily_returns = prices_df["return"] / 100  # Convert percentage to decimal

        # Expected annual return using daily returns
        expected_annual_return = (1 + daily_returns.mean()) ** 250 - 1

        # Calculate Sharpe Ratio
        sharpe_ratio = (((expected_annual_return - risk_free_rate) / prices_df["return"].std()) * np.sqrt(250)).round(3)

        cum_return = (np.log(prices_df['Total Value'].iloc[-1]) - np.log(prices_df['Total Value'].iloc[0])) * 100

        # Prepare risk and Sharpe ratio data for table
        risk_data = pd.DataFrame({
            'Risk': individual_risks,
            'Sharpe Ratio': sharpe_individual
        }).sort_values(by='Risk', ascending=False)

        risk_table = risk_data.reset_index().rename(columns={'index': 'Ticker'}).to_dict('records')

        # Calculate different time period returns for each stock
        # Get dates for different time periods
        today = datetime.now()
        ytd_start = datetime(today.year, 1, 1)
        one_year_ago = today - timedelta(days=365)
        three_years_ago = today - timedelta(days=365 * 3)
        five_years_ago = today - timedelta(days=365 * 5)

        # Convert dates to string format for indexing
        ytd_start_str = ytd_start.strftime('%Y-%m-%d')
        one_year_ago_str = one_year_ago.strftime('%Y-%m-%d')
        three_years_ago_str = three_years_ago.strftime('%Y-%m-%d')
        five_years_ago_str = five_years_ago.strftime('%Y-%m-%d')

        # Initialize DataFrames to store returns
        returns_data = pd.DataFrame(index=stock_list)

        # Calculate returns for each period
        for ticker in stock_list:
            # Get the price series for the stock
            price_series = df_prices[ticker]

            # Get the latest price
            latest_price = price_series.iloc[-1]

            # YTD return
            ytd_idx = price_series.index[price_series.index >= ytd_start_str]
            if len(ytd_idx) > 0:
                ytd_price = price_series.loc[ytd_idx[0]]
                returns_data.loc[ticker, 'YTD'] = ((latest_price / ytd_price) - 1) * 100
            else:
                returns_data.loc[ticker, 'YTD'] = None

            # 1-year return
            one_year_idx = price_series.index[price_series.index >= one_year_ago_str]
            if len(one_year_idx) > 0:
                one_year_price = price_series.loc[one_year_idx[0]]
                returns_data.loc[ticker, '1 Year'] = ((latest_price / one_year_price) - 1) * 100
            else:
                returns_data.loc[ticker, '1 Year'] = None

            # 3-year return
            three_year_idx = price_series.index[price_series.index >= three_years_ago_str]
            if len(three_year_idx) > 0:
                three_year_price = price_series.loc[three_year_idx[0]]
                returns_data.loc[ticker, '3 Year'] = ((latest_price / three_year_price) - 1) * 100
            else:
                returns_data.loc[ticker, '3 Year'] = None

            # 5-year return
            five_year_idx = price_series.index[price_series.index >= five_years_ago_str]
            if len(five_year_idx) > 0:
                five_year_price = price_series.loc[five_year_idx[0]]
                returns_data.loc[ticker, '5 Year'] = ((latest_price / five_year_price) - 1) * 100
            else:
                returns_data.loc[ticker, '5 Year'] = None

        # Prepare returns table in the same order as risk_table
        ticker_order = [item['Ticker'] for item in risk_table]
        returns_data = returns_data.loc[ticker_order]
        returns_table = returns_data.reset_index().rename(columns={'index': 'Ticker'}).to_dict('records')

        return render_template('analysis.html',
                               has_data=True,
                               sd_p_annual=sd_p_annual * 100,
                               sharpe_ratio=sharpe_ratio,
                               cum_return=cum_return,
                               risk_table=risk_table,
                               returns_table=returns_table)
    except Exception as e:
        print(f"Error in analysis: {e}")
        return render_template('analysis.html', has_data=False, error=str(e))


if __name__ == '__main__':
    app.run(debug=True)