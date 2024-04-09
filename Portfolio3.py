import datetime
import pandas as pd
import yahooquery as yq
import sqlite3
import warnings
import os
import functools
import time
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

    def add_past_position(self, df, purchase_date: str):
        '''This takes a data frame of symbols, direction, shares, along with a purchase date and adds it to the portfolio'''
        
        # Ensure the DataFrame has the correct columns
        if set(df.columns) != set(['symbol', 'direction', 'shares']):
            raise ValueError('DataFrame must have columns: symbol, direction, shares')
        
        pricedict = self._createPriceDict(df, 'open', purchase_date)

        # I am going to be slightly lazy and just insert one by one
        for _, trade in df.iterrows():
            symbol = trade['symbol']
            purchase_price = pricedict[symbol]

            self.add_position_single(symbol, trade['direction'], trade['shares'], purchase_price, purchase_date)

    def _profitPerTrade(self, trade, pricedict, builtin: str = 'pershare', ppt_func=None):
        # the ppt_func, takes ppt, purchase_price, num_shares as arguments and by default returns ppt, however you can adapt as you wish
        
        symbol = trade['symbol']
        num_shares = trade['shares']
        direction = trade['direction']
        purchase_price = trade['purchase_price']

        if direction == 'long':
            ppt = pricedict[symbol] - purchase_price
        else:
            ppt = purchase_price - pricedict[symbol]

        if builtin == 'pershare':
            return ppt * num_shares
        
        elif builtin == 'percentage':
            return num_shares * ppt / purchase_price
        
        elif builtin == 'custom':
            if ppt_func is None:
                raise ValueError('ppt_func must be a function')
            return ppt_func(ppt, purchase_price, num_shares)

    def _createPriceDict(self, df, type: str, date: str = None):
        '''This function will create a dictionary of the most recent prices for each symbol in the portfolio, returns price dict. Type must be either 'close' or 'current' '''
        # For hashing we want the list to always be sorted, but we require it to be immutable
        symbols = tuple(sorted(list(set(df['symbol'].to_list()))))
        return self._createPriceDictCache(symbols, type, date)

    @functools.lru_cache(maxsize=128)
    def _createPriceDictCache(self, symbols: tuple, type: str, date=None):
        ''' This function uses the lru_cache to store the most recent prices for each symbol in the portfolio, returns price dict. Type must be either 'close' or 'current' '''
        # We briefly required symbols to be immutable, now we convert back to a list for the function
        symbols = list(symbols)
        
        if symbols == []:
            raise ValueError('No symbols in the portfolio')
        
        pricedict = {}
        dead_symbols = []
        
        if type == 'close' or type == 'open':
            data = yq.Ticker(symbols).history(period='1d', start=date)
            recentClose = data[type]

            for symbol in symbols:
                if symbol in recentClose:
                    pricedict[symbol] = recentClose[symbol][0]
                else:
                    print(f'{symbol} not in data')
                    dead_symbols.append(symbol)
            for sym in dead_symbols:
                symbols.remove(sym)
        
        elif type == 'current':
            data = yq.Ticker(symbols).history(period='1d', interval='1m')

            for symbol in symbols:
                if symbol in data.index:
                    pricedict[symbol] = data.loc[(symbol, slice(None))]['close'][-1]
                else:
                    print(f'{symbol} not in data.')
                    dead_symbols.append(symbol)
            for sym in dead_symbols:
                symbols.remove(sym)

        else:
            raise ValueError('type must be either "close", "open" or "current"')

        return pricedict

    def initalValue(self):
        '''Get the initial value of the portfolio'''

        portfolio_df = pd.read_sql('SELECT shares, purchase_price FROM portfolio', self.conn)
        each_trade_val = portfolio_df['shares'] * portfolio_df['purchase_price']
        return sum(each_trade_val)
    
    def getValue(self, time_type: str = None, profit_type: str = None, date: str = None):
        '''This will get either the close value, at a given date, or the current value. Giving either as absolute or percentage profit.'''
        
        if profit_type not in ['percentage', 'absolute']:
            raise ValueError('profit_type must be either "percentage" or "absolute"')
        
        portfolio_df = pd.read_sql('SELECT * FROM portfolio', self.conn)

        if time_type == 'current':
            pricedict = self._createPriceDict(portfolio_df, 'current')
        elif time_type == 'close':
            pricedict = self._createPriceDict(portfolio_df, 'close', date)
        else:
            raise ValueError('time_type must be either "current" or "close"')

        running_calculation = 0

        for _, trade in portfolio_df.iterrows():
            running_calculation += self._profitPerTrade(trade, pricedict, 'absolute')
        
        if profit_type == 'percentage':
            return running_calculation / self.initalValue()
        else:
            return running_calculation 
        
    def absoluteProfit(self, time_type: str = None, date: str = None):
        return self.getValue(time_type, 'absolute', date)

    def percentageProfit(self, time_type: str = None, date: str = None):
        return self.getValue(time_type, 'percentage', date)
        
    def trade_sell_after(self, days: int):
        ''' Calculates the profit of selling each position after a days amount of days after it was purchased. I am not sure how to handle weekends yet.'''
        df = pd.read_sql('SELECT * FROM portfolio', self.conn)
        # Convert the purchase date to a datetime object
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
                portfolio_value += self._profitPerTrade(trade, pricedict, 'percentage')

        return portfolio_value
    
    def upDown(self, verbose: bool = False):
        '''This function will return the percentage of the portfolio as well a list of what is up and  that is up and the percentage that is down'''
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

        if verbose:
            retstr = f'''Percentage of positions profiting: {len(up) / num_positions * 100}%
            Positions that are up: {up}
            Positions that are down: {down}'''
        else:
            retstr = f'''Percentage of positions profiting: {len(up) / num_positions * 100}%'''

        return retstr
    
        
    def generate_test_data(self):
        # Generate some test data for the portfolio
        test_data = [
            ('AAPL', 'long', 1, 0, '2021-01-01'),
            ('GOOGL', 'long', 1, 0, '2021-02-01'),
            ('MSFT', 'long', 1, 0, '2021-03-01'),
            ('AMZN', 'long', 1, 0, '2021-04-01'),
        ]

        for data in test_data:
            self.add_position_single(*data)
    
    def close(self):
        self.conn.close()

    def __del__(self):
        if self.conn:
            self.conn.close()

        
if __name__ == '__main__':
    port = Portfolio(r'C:\Users\chris\Documents\Options\PostSQL\MyPortfolios\FixedMultipleSkewPortfolio.db')
    #df = pd.DataFrame({'symbol': ['AAPL', 'GOOGL', 'MSFT', 'AMZN'], 'direction': ['long', 'long', 'long', 'long'], 'shares': [1, 1, 1, 1]})
    print(port.trade_sell_after(3))
    
