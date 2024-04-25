import datetime
import pandas as pd
import yahooquery as yq
import sqlite3
import warnings
import os
import functools
import time
import logging 
logging.basicConfig(level=logging.WARNING) 

warnings.filterwarnings('ignore')
# In this version I have made getting profits more streamlined

# You should optimse where unessecary calls are not made, for example if you have already fetched the prices there is no need to fetch them again.

class Portfolio:
    def __init__(self, db_file):
        self.db_file = db_file
        
        if not os.path.exists(db_file):
            print('Creating new database.')
            self.conn = sqlite3.connect(db_file)
            self.cursor = self.conn.cursor()
            self.create_table()

        else:
            self.conn = sqlite3.connect(db_file)
            # Add functionaility which checks table is of the same format.
            self.cursor = self.conn.cursor()
    
    def create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio (
                symbol TEXT,
                direction TEXT,
                shares INTEGER,
                purchase_price REAL,
                purchase_date TEXT
            )
        ''')
        self.conn.commit()

    def __str__(self):
        return str(pd.read_sql('SELECT * FROM portfolio', self.conn))

    def add_position_single(self, symbol, direction, shares, purchase_price, purchase_date):
        self.cursor.execute('''
            INSERT INTO portfolio (symbol, direction, shares, purchase_price, purchase_date)
            VALUES (?, ?, ?, ?, ?)
        ''', (symbol, direction, shares, purchase_price, purchase_date))
        self.conn.commit()
    
    def add_position_df(self, df):
        # Ensure the DataFrame has the correct columns
        if set(df.columns) != set(['symbol', 'direction', 'shares', 'purchase_price', 'purchase_date']):
            raise ValueError('DataFrame must have columns: symbol, direction, shares, purchase_price, purchase_date')

        df.to_sql('portfolio', self.conn, if_exists='append', index=False)

    def add_past_position(self, df, purchase_date: str, type: str = 'open'):
        '''This takes a data frame of symbols, direction, shares, along with a purchase date and adds it to the portfolio'''
        
        assert type in ['open', 'close', 'current'], 'type must be either "close", "open" or "current"'

        if set(df.columns) != set(['symbol', 'direction', 'shares']):   # Ensure the DataFrame has the correct columns
            raise ValueError('DataFrame must have columns: symbol, direction, shares')
        
        pricedict = self._createPriceDict(df, type, purchase_date)

        for _, trade in df.iterrows():  # I am going to be slightly lazy and just insert one by one
            symbol = trade['symbol']
            purchase_price = pricedict[symbol]

            self.add_position_single(symbol, trade['direction'], trade['shares'], purchase_price, purchase_date)
        
    def _profitPerTrade(self, trade, pricedict, pershare=True):
        '''This function will return the profit per trade, given the trade, pricedict, and the type of profit. The profit type can be either 'pershare', 'percentage', or 'custom'.
        If custom is chosen, a function must be passed to ppt_func, which will take ppt, purchase_price, num_shares as arguments and return the profit. '''
        
        symbol = trade['symbol']
        num_shares = trade['shares']
        direction = trade['direction']
        purchase_price = trade['purchase_price']

        if direction == 'long':
            ppt = pricedict[symbol] - purchase_price
        else:
            ppt = purchase_price - pricedict[symbol]

        if pershare:
            return ppt * num_shares
        else:
            return num_shares * ppt / purchase_price

    def _createPriceDict(self, df, type: str, date: str = None):
        '''This function will create a dictionary of the most recent prices for each symbol in the portfolio, returns price dict. Type must be either 'close' or 'current' '''
        assert type in ['open', 'close', 'current'], 'type must be either "close", "open" or "current"'
        
        symbols = tuple(sorted(list(set(df['symbol'].to_list())))) # For hashing we want the list to always be sorted, but we require it to be immutable
        return self._createPriceDictCache(symbols, type, date)

    @functools.lru_cache(maxsize=128)
    def _createPriceDictCache(self, symbols: tuple, type: str, date=None):
        ''' This function uses the lru_cache to store the most recent prices for each symbol in the portfolio, returns price dict. Type must be either 'close' or 'current' '''

        symbols = list(symbols) # We briefly required symbols to be immutable, now we convert back to a list for the function
        assert symbols != [], 'No symbols in the portfolio'
        
        pricedict = {}
        dead_symbols = []
        
        if type == 'close' or type == 'open':
            data = yq.Ticker(symbols).history(period='1d', start=date)
            recentClose = data[type]

            for symbol in symbols:
                if symbol in recentClose:
                    pricedict[symbol] = recentClose[symbol][0]
                else:
                    logging.warning(f'{symbol} not in data')
                    dead_symbols.append(symbol)
            for sym in dead_symbols:
                symbols.remove(sym)
        
        elif type == 'current':
            data = yq.Ticker(symbols).history(period='1d', interval='1m')

            for symbol in symbols:
                if symbol in data.index:
                    pricedict[symbol] = data.loc[(symbol, slice(None))]['close'][-1]
                else:
                    logging.warning(f'{symbol} not in data')
                    dead_symbols.append(symbol)
            for sym in dead_symbols:
                symbols.remove(sym)

        return pricedict

    def initalValue(self):
        '''Get the initial value of the portfolio'''

        portfolio_df = pd.read_sql('SELECT shares, purchase_price FROM portfolio', self.conn)
        each_trade_val = portfolio_df['shares'] * portfolio_df['purchase_price']
        return sum(each_trade_val)
    
        
    def absoluteProfit(self, time_type: str = 'current', date: str = None):
        ''' I dont really like the new way of doing the percentage profit, I think it is a bit too complicated. 
            And unnintuative Time type should be either 'current', 'close', or 'open' and date should be a string in the format 'YYYY-MM-DD' '''

        portfolio_df = pd.read_sql('SELECT * FROM portfolio', self.conn)
        pricedict = self._createPriceDict(portfolio_df, time_type, date)

        running_calculation = 0
        for _, trade in portfolio_df.iterrows():
            prof = self._profitPerTrade(trade, pricedict, 'pershare')
            running_calculation += prof

        return running_calculation 
        

    def percentageProfit(self, time_type: str = 'current', date: str = None):
        ''' I dont really like the new way of doing the percentage profit, I think it is a bit too complicated. 
            And unnintuative Time type should be either 'current', 'close', or 'open' and date should be a string in the format 'YYYY-MM-DD' '''

        portfolio_df = pd.read_sql('SELECT * FROM portfolio', self.conn)
        pricedict = self._createPriceDict(portfolio_df, time_type, date)

        running_calculation = 0
        for _, trade in portfolio_df.iterrows():
            prof = self._profitPerTrade(trade, pricedict, 'pershare')
            running_calculation += prof

        return running_calculation / self.initalValue()
        
        
    def trade_sell_after(self, days: int):
        ''' Calculates the profit of selling each position after a days amount of days after it was purchased. I am not sure how to handle weekends yet.'''
        
        df = pd.read_sql('SELECT * FROM portfolio', self.conn)
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        
        # We now get the the different purchase dates, filter by the ones that are less than days away from the current date
        purchase_dates = df['purchase_date'].to_frame('purchase_date')
        purchase_dates['purchase_date'] = pd.to_datetime(purchase_dates['purchase_date'])
        purchase_dates = purchase_dates.drop_duplicates()
        purchase_dates['sell_date'] = purchase_dates['purchase_date'] + pd.DateOffset(days=days)
        final_purchase_dates = purchase_dates[purchase_dates['sell_date'] <= pd.Timestamp.now()]
        
        portfolio_value = 0
        for _, (purchase_date, sell_date) in final_purchase_dates.iterrows():
            purchase_date_df = df[df['purchase_date'] == purchase_date]
            pricedict = self._createPriceDict(purchase_date_df, 'close', sell_date)
            for _, trade in purchase_date_df.iterrows():
                portfolio_value += self._profitPerTrade(trade, pricedict)

        return portfolio_value / self.initalValue()
    
    def volatility(self):
        '''This function will return the volatility of the portfolio'''

        df = pd.read_sql('SELECT * FROM portfolio', self.conn)

        dates = pd.to_datetime(df['purchase_date'])
        dates = dates.sort_values()

        raw_daily_returns = [self.absoluteProfit('close', date) for date in dates]
        daily_returns = [raw_daily_returns[i] - raw_daily_returns[i-1] for i in range(1, len(raw_daily_returns))]

        return pd.Series(daily_returns).std()
    
    def sharpe_ratio(self, risk_free_rate: float = 0.02):
        '''This function will return the sharpe ratio of the portfolio, the risk free rate should be in decimal form'''
        return (self.percentageProfit() - risk_free_rate) / self.volatility()
    
    
    def upDown(self, verbose: bool = False):
        '''This function will return the percentage of the portfolio that is up and the percentage that is down'''
        df = pd.read_sql('SELECT * FROM portfolio', self.conn)
        pricedict = self._createPriceDict(df, 'current')

        num_positions = len(df)
        up, down = [], []

        for _, trade in df.iterrows():
            pps = self._profitPerTrade(trade, pricedict)
            if pps >= 0:
                up.append(trade)
            else:
                down.append(trade)

        percentage_up = len(up) / num_positions * 100

        if verbose:
            retstr = f'''Percentage of positions profiting: {percentage_up:.2f}% 
            Positions that are up: {", ".join([str(trade['symbol']) for trade in up])[:100]}
            Positions that are down: {", ".join([str(trade['symbol']) for trade in down])[:100]}'''
        else:
            retstr = f'''Percentage of positions profiting: {percentage_up:.2f}%'''

        return retstr
    
    def generate_test_data(self):
        '''This function will generate some test data for the portfolio'''
        test_data = [
            ('AAPL', 'long', 1, 100, '2021-01-01'),
            ('GOOGL', 'long', 1, 300, '2021-02-01'),
            ('MSFT', 'long', 1, 150, '2021-03-01'),
            ('AMZN', 'long', 1, 20, '2021-04-01'),
        ]

        self.add_position_df(pd.DataFrame(test_data, columns=['symbol', 'direction', 'shares', 'purchase_price', 'purchase_date']))

    def __del__(self):
        if self.conn:
            self.conn.close()