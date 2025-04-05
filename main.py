from fastapi import FastAPI, Query, HTTPException
import yfinance as yf
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import requests
import time
from functools import lru_cache
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache to store responses and avoid redundant API calls
@lru_cache(maxsize=1000)
def get_stock_data(ticker: str, period: str = "1d", interval: str = "1h") -> Optional[pd.DataFrame]:
    """
    Fetch stock data with caching
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        if data.empty:
            logger.warning(f"No data found for {ticker}")
            return None
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def get_multiple_stocks(tickers: List[str], period: str = "1d", interval: str = "1h") -> Optional[Dict[str, pd.DataFrame]]:
    """
    Fetch multiple stocks efficiently
    """
    try:
        # Use yf.download for batch fetching
        data = yf.download(" ".join(tickers), period=period, interval=interval, group_by='ticker')
        if data.empty:
            logger.warning("No data found for any of the tickers")
            return None
            
        results = {}
        for ticker in tickers:
            if len(tickers) > 1:
                if ticker in data.columns.levels[1]:
                    results[ticker] = data[ticker]
            else:
                results[ticker] = data
                
        return results
    except Exception as e:
        logger.error(f"Error fetching multiple stocks: {str(e)}")
        return None

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Financial Data API",
    description="Comprehensive API for stocks, forex, mutual funds, and index funds data",
    version="1.0.0"
)

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants for time periods
TIME_PERIODS = {
    "30d": 30,
    "90d": 90,
    "1y": 365,
    "all": None  # None will use max available data in yfinance
}

# Exchange suffix mapping
EXCHANGE_MAPPING = {
    "nse": ".NS",
    "bse": ".BO",
    "nasdaq": "",
    "nyse": ""
}

# Common indices mapping for easy reference
INDICES_MAPPING = {
    "nifty50": "^NSEI",
    "sensex": "^BSESN",
    "sp500": "^GSPC",
    "dow": "^DJI",
    "nasdaq": "^IXIC",
    "ftse": "^FTSE",
    "nikkei": "^N225"
}

# Forex pairs mapping
FOREX_MAPPING = {
    "usd_inr": "USD/INR=X",
    "eur_inr": "EUR/INR=X",
    "gbp_usd": "GBP/USD=X",
    "eur_usd": "EUR/USD=X",
    "jpy_usd": "JPY/USD=X"
}

# Helper function to calculate date range from period
def get_date_range(period: str) -> tuple[Optional[datetime], datetime]:
    days = TIME_PERIODS.get(period, 90)
    end_date = datetime.now()
    start_date = None if days is None else end_date - timedelta(days=days)
    return start_date, end_date

# Helper function to calculate beta
def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Calculate beta using covariance of stock returns with market returns
    """
    covariance = stock_returns.cov(market_returns)
    market_variance = market_returns.var()
    return float(covariance / market_variance) if market_variance != 0 else None

# Helper function to calculate alpha
def calculate_alpha(stock_returns: pd.Series, market_returns: pd.Series, risk_free_rate: float = 0.04) -> float:
    """
    Calculate alpha (excess return) using the CAPM model
    risk_free_rate is annualized (default 4%)
    """
    # Get actual number of trading days in the data
    n_trading_days = len(stock_returns)
    
    # Convert annual risk-free rate to daily
    daily_rf = (1 + risk_free_rate) ** (1/n_trading_days) - 1
    
    # Calculate average returns using actual trading days
    avg_stock_return = stock_returns.mean() * n_trading_days  # Annualized
    avg_market_return = market_returns.mean() * n_trading_days  # Annualized
    
    # Calculate beta
    beta = calculate_beta(stock_returns, market_returns)
    if beta is None:
        return None
        
    # Calculate alpha
    alpha = avg_stock_return - (risk_free_rate + beta * (avg_market_return - risk_free_rate))
    return float(alpha)

# Helper function to calculate maximum drawdown
def calculate_max_drawdown(prices: pd.Series) -> float:
    """
    Calculate maximum drawdown from peak to trough
    """
    # Calculate cumulative maximum
    rolling_max = prices.expanding().max()
    # Calculate drawdowns
    drawdowns = prices / rolling_max - 1
    # Get maximum drawdown
    max_drawdown = drawdowns.min()
    return float(max_drawdown) if not pd.isna(max_drawdown) else None

# Helper function to calculate Sharpe ratio
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.04) -> float:
    """
    Calculate Sharpe ratio using daily returns
    risk_free_rate is annualized (default 4%)
    """
    # Convert annual risk-free rate to daily
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    
    # Calculate excess returns
    excess_returns = returns - daily_rf
    
    # Calculate annualized metrics
    annualized_return = excess_returns.mean() * 252
    annualized_volatility = returns.std() * np.sqrt(252)
    
    # Calculate Sharpe ratio
    sharpe = annualized_return / annualized_volatility if annualized_volatility != 0 else None
    return float(sharpe) if sharpe is not None else None

# Helper function to calculate trust score
def calculate_trust_score(info: Dict[str, Any], std_dev: float, max_dd: float, beta: float, sharpe: float) -> float:
    """
    Calculate trust score (1-5) based on various metrics
    """
    score = 3.0  # Start with neutral score
    
    # Market cap factor (larger is better)
    market_cap = info.get('marketCap', 0)
    if market_cap > 10e9:  # > $10B
        score += 0.5
    elif market_cap > 1e9:  # > $1B
        score += 0.3
    elif market_cap < 100e6:  # < $100M
        score -= 0.5
    
    # Volume factor (higher is better)
    volume = info.get('volume', 0)
    if volume > 1e6:
        score += 0.3
    elif volume < 100e3:
        score -= 0.3
    
    # Volatility factor (lower is better)
    if std_dev < 0.2:
        score += 0.4
    elif std_dev > 0.4:
        score -= 0.4
    
    # Drawdown factor (less negative is better)
    if max_dd > -0.1:
        score += 0.3
    elif max_dd < -0.3:
        score -= 0.3
    
    # Beta factor (closer to 1 is better)
    if 0.8 <= beta <= 1.2:
        score += 0.3
    elif beta > 1.5 or beta < 0.5:
        score -= 0.3
    
    # Sharpe ratio factor (higher is better)
    if sharpe > 1.0:
        score += 0.3
    elif sharpe < 0:
        score -= 0.3
    
    # Ensure score is between 1 and 5
    return max(1.0, min(5.0, score))

# Helper function to determine safety level
def determine_safety_level(trust_score: float) -> str:
    """
    Determine safety level based on trust score
    """
    if trust_score >= 4.0:
        return "High"
    elif trust_score >= 2.5:
        return "Mid"
    else:
        return "Low"

# Helper function to calculate annual return
def calculate_annual_return(data: pd.DataFrame) -> float:
    """
    Calculate annual return from historical data
    """
    if len(data) < 2:
        return 0.0
    
    # Calculate total return over the period
    start_price = data['Close'].iloc[0]
    end_price = data['Close'].iloc[-1]
    total_return = (end_price / start_price) - 1
    
    # Annualize the return
    # Ensure we're working with timezone-naive datetime objects
    start_date = data.index[0]
    end_date = data.index[-1]
    
    if start_date.tz is not None:
        start_date = start_date.tz_localize(None)
    if end_date.tz is not None:
        end_date = end_date.tz_localize(None)
        
    days = (end_date - start_date).days
    if days > 0:
        annual_return = (1 + total_return) ** (365 / days) - 1
    else:
        annual_return = 0.0
        
    return annual_return

# Helper function to format stock data
def format_stock_data(data: pd.DataFrame, info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format stock data for API response
    """
    # Ensure data index is timezone-naive
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    
    # Calculate daily returns
    data['Returns'] = data['Close'].pct_change()
    
    # Calculate one-year standard deviation
    one_year_ago = datetime.now().replace(tzinfo=None) - timedelta(days=365)
    one_year_data = data[data.index >= one_year_ago]
    std_dev = one_year_data['Returns'].std()
    
    # Calculate maximum drawdown
    max_dd = calculate_max_drawdown(data['Close'])
    
    # Calculate Sharpe ratio
    sharpe = calculate_sharpe_ratio(one_year_data['Returns'])
    
    # Calculate beta and alpha
    try:
        # Get market data (S&P 500)
        market = yf.Ticker("^GSPC")
        market_data = market.history(start=one_year_ago)
        
        # Ensure market data index is timezone-naive
        if market_data.index.tz is not None:
            market_data.index = market_data.index.tz_localize(None)
            
        market_data['Returns'] = market_data['Close'].pct_change()
        
        # Align dates and calculate beta and alpha
        aligned_data = pd.concat([
            one_year_data['Returns'].rename('stock_returns'),
            market_data['Returns'].rename('market_returns')
        ], axis=1).dropna()
        
        beta = calculate_beta(
            aligned_data['stock_returns'],
            aligned_data['market_returns']
        )
        
        alpha = calculate_alpha(
            aligned_data['stock_returns'],
            aligned_data['market_returns']
        )
    except Exception as e:
        logger.error(f"Error calculating beta/alpha: {str(e)}")
        beta = None
        alpha = None
    
    # Calculate trust score and safety level
    trust_score = calculate_trust_score(info, std_dev, max_dd, beta or 1.0, sharpe or 0.0)
    safety_level = determine_safety_level(trust_score)
    
    # Calculate annual return
    annual_return = calculate_annual_return(data)
    
    return {
        "Open": data['Open'].tolist(),
        "High": data['High'].tolist(),
        "Low": data['Low'].tolist(),
        "Close": data['Close'].tolist(),
        "Volume": data['Volume'].tolist() if 'Volume' in data.columns else None,
        "Dates": data.index.strftime('%Y-%m-%d').tolist(),
        "Returns": data['Returns'].tolist(),
        "OneYearStandardDeviation": float(std_dev) if not pd.isna(std_dev) else None,
        "Beta": beta,
        "Alpha": alpha,
        "MaximumDrawdown": max_dd,
        "SharpeRatio": sharpe,
        "TrustScore": float(trust_score),
        "SafetyLevel": safety_level,
        "AnnualReturn": float(annual_return)
    }

# ✅ 1️⃣ Root Endpoint (Check if API is running)
@app.get("/")
def root():
    return {
        "message": "Welcome to the Enhanced Financial Data API!",
        "version": "1.0.0",
        "features": [
            "Stock Data (Indian & Global)",
            "Forex Trading Data",
            "Index Fund Data",
            "Mutual Fund Data"
        ],
        "endpoints": [
            "/search/",
            "/stock/{symbol}",
            "/forex/{pair}",
            "/index/{symbol}",
            "/mutual-fund/{symbol}",
            "/compare/",
            "/mutual-funds/india/popular"
        ]
    }

# ✅ 2️⃣ Search Stocks by Name or Symbol
@app.get("/search/")
def search_instruments(
    query: str, 
    limit: Optional[int] = Query(10, description="Maximum number of results"),
    type: Optional[str] = Query(None, description="Filter by type: stock, forex, index, mutual_fund")
):
    try:
        results = []
        
        # Add stock search
        if type is None or type == "stock":
            # Try to get stock info directly
            try:
                stock = yf.Ticker(query)
                info = stock.info
                if info and info.get('quoteType') == 'EQUITY':
                    exchange = ""
                    if ".NS" in query:
                        exchange = "NSE"
                    elif ".BO" in query:
                        exchange = "BSE"
                    elif info.get('exchange') in ['NMS', 'NGS', 'NCM', 'NGM']:
                        exchange = "NASDAQ"
                    elif info.get('exchange') in ['NYQ', 'PSE', 'PCX', 'ASE', 'AMX']:
                        exchange = "NYSE"
                        
                    results.append({
                        "symbol": query,
                        "name": info.get('shortName', 'Unknown'),
                        "exchange": exchange,
                        "type": "stock"
                    })
            except:
                pass
        
        # Add forex search
        if type is None or type == "forex":
            for key, symbol in FOREX_MAPPING.items():
                if query.lower() in key or query.lower() in symbol.lower():
                    results.append({
                        "symbol": symbol,
                        "name": symbol.replace("=X", ""),
                        "type": "forex"
                    })
        
        # Add index search
        if type is None or type == "index":
            for key, symbol in INDICES_MAPPING.items():
                if query.lower() in key or query.lower() in symbol.lower():
                    results.append({
                        "symbol": symbol,
                        "name": key.capitalize(),
                        "type": "index"
                    })
        
        return {"results": results[:limit]}
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during search")

# ✅ 3️⃣ Get Stock Data (with exchange support)
@app.get("/stock/{symbol}")
def get_stock_data_endpoint(
    symbol: str,
    period: Optional[str] = Query("90d", description="Time period: 30d, 90d, 1y, all"),
    interval: Optional[str] = Query("1d", description="Data interval: 1d, 1wk, 1mo"),
    exchange: Optional[str] = Query(None, description="Exchange: nse, bse")
):
    try:
        # Add exchange suffix if specified
        if exchange and exchange.lower() in EXCHANGE_MAPPING:
            if not symbol.endswith(EXCHANGE_MAPPING[exchange.lower()]):
                symbol = symbol + EXCHANGE_MAPPING[exchange.lower()]
        
        # Get stock info first
        stock = yf.Ticker(symbol)
        info = stock.info
        
        if not info:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
        
        # Get historical data
        data = get_stock_data(symbol, period=period, interval=interval)
        if data is None:
            start_date, end_date = get_date_range(period)
            data = stock.history(start=start_date, end=end_date, interval=interval)
            
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for {symbol}")
        
        # Format response
        return {
            "symbol": symbol,
            "name": info.get('shortName', 'Unknown'),
            "sector": info.get('sector', 'Unknown'),
            "industry": info.get('industry', 'Unknown'),
            "market_cap": info.get('marketCap', 'Unknown'),
            "current_price": info.get('currentPrice', info.get('regularMarketPrice', None)),
            "previous_close": info.get('previousClose', None),
            "day_high": info.get('dayHigh', None),
            "day_low": info.get('dayLow', None),
            "volume": info.get('volume', None),
            "period": period,
            "interval": interval,
            "data": format_stock_data(data, info)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching stock data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while fetching stock data")

# ✅ 4️⃣ Get Forex Data
@app.get("/forex/{pair}")
def get_forex_data(
    pair: str,
    period: Optional[str] = Query("90d", description="Time period: 30d, 90d, 1y, all"),
    interval: Optional[str] = Query("1d", description="Data interval: 1d, 1wk, 1mo")
):
    try:
        # Map common forex pair names
        symbol = FOREX_MAPPING.get(pair.lower(), pair)
        if not symbol.endswith('=X'):
            symbol = symbol + '=X'
            
        forex = yf.Ticker(symbol)
        info = forex.info
        
        if not info:
            raise HTTPException(status_code=404, detail=f"Forex pair {pair} not found")
        
        # Get historical data
        data = get_stock_data(symbol, period=period, interval=interval)
        if data is None:
            start_date, end_date = get_date_range(period)
            data = forex.history(start=start_date, end=end_date, interval=interval)
            
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for {pair}")
        
        # Format response
        return {
            "symbol": symbol,
            "name": info.get('shortName', symbol),
            "current_rate": info.get('regularMarketPrice', None),
            "previous_close": info.get('previousClose', None),
            "day_high": info.get('dayHigh', None),
            "day_low": info.get('dayLow', None),
            "period": period,
            "interval": interval,
            "data": format_stock_data(data, info)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching forex data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while fetching forex data")

# ✅ 5️⃣ Get Index Data
@app.get("/index/{symbol}")
def get_index_data(
    symbol: str,
    period: Optional[str] = Query("90d", description="Time period: 30d, 90d, 1y, all"),
    interval: Optional[str] = Query("1d", description="Data interval: 1d, 1wk, 1mo")
):
    # Map from common names to actual symbols
    actual_symbol = INDICES_MAPPING.get(symbol.lower(), symbol)
    
    try:
        index = yf.Ticker(actual_symbol)
        info = index.info
        
        if not info:
            raise HTTPException(status_code=404, detail=f"Index {symbol} not found")
        
        # Get historical data
        data = get_stock_data(actual_symbol, period=period, interval=interval)
        if data is None:
            start_date, end_date = get_date_range(period)
            data = index.history(start=start_date, end=end_date, interval=interval)
            
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for {symbol}")
        
        # Format response
        return {
            "symbol": actual_symbol,
            "name": info.get('shortName', actual_symbol),
            "current_value": info.get('regularMarketPrice', None),
            "previous_close": info.get('previousClose', None),
            "day_high": info.get('dayHigh', None),
            "day_low": info.get('dayLow', None),
            "period": period,
            "interval": interval,
            "data": format_stock_data(data, info)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching index data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while fetching index data")

# ✅ 6️⃣ Get Mutual Fund Data
@app.get("/mutual-fund/{symbol}")
def get_mutual_fund_data(
    symbol: str,
    period: Optional[str] = Query("90d", description="Time period: 30d, 90d, 1y, all"),
    interval: Optional[str] = Query("1d", description="Data interval: 1d, 1wk, 1mo")
):
    try:
        fund = yf.Ticker(symbol)
        info = fund.info
        
        if not info:
            raise HTTPException(status_code=404, detail=f"Mutual fund {symbol} not found")
        
        # Get historical data
        data = get_stock_data(symbol, period=period, interval=interval)
        if data is None:
            start_date, end_date = get_date_range(period)
            data = fund.history(start=start_date, end=end_date, interval=interval)
            
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for {symbol}")
        
        # Format response
        return {
            "symbol": symbol,
            "name": info.get('shortName', 'Unknown'),
            "category": info.get('category', 'Unknown'),
            "current_nav": info.get('regularMarketPrice', None),
            "previous_nav": info.get('previousClose', None),
            "day_high": info.get('dayHigh', None),
            "day_low": info.get('dayLow', None),
            "period": period,
            "interval": interval,
            "data": format_stock_data(data, info)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching mutual fund data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while fetching mutual fund data")

# ✅ 7️⃣ Compare Multiple Instruments
@app.get("/compare/")
def compare_instruments(
    symbols: str = Query(..., description="Comma-separated list of symbols to compare"),
    period: Optional[str] = Query("90d", description="Time period: 30d, 90d, 1y, all"),
    interval: Optional[str] = Query("1d", description="Data interval: 1d, 1wk, 1mo")
):
    symbol_list = [s.strip() for s in symbols.split(',')]
    
    try:
        # Get date range
        start_date, end_date = get_date_range(period)
        
        # Use the optimized get_multiple_stocks function for batch fetching
        if len(symbol_list) > 1:
            result = get_multiple_stocks(symbol_list, period=period, interval=interval)
            
            if not result:
                # Fallback to yf.download
                data = yf.download(symbol_list, start=start_date, end=end_date, interval=interval)
                
                if data.empty:
                    raise HTTPException(status_code=404, detail="No data found for the specified symbols")
                
                # Format the multi-level columns for easier consumption
                result = {}
                for field in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if field in data.columns.levels[0]:
                        field_data = data[field].reset_index()
                        result[field] = field_data.to_dict(orient="records")
                
                return {
                    "symbols": symbol_list,
                    "period": period,
                    "interval": interval,
                    "data": result
                }
            else:
                # Format the data from get_multiple_stocks
                formatted_result = {}
                for ticker, ticker_data in result.items():
                    formatted_result[ticker] = format_stock_data(ticker_data, {})
                
                return {
                    "symbols": symbol_list,
                    "period": period,
                    "interval": interval,
                    "data": formatted_result
                }
        else:
            # For single symbol, reuse get_stock_data
            data = get_stock_data(symbol_list[0], period=period, interval=interval)
            if data is None:
                data = yf.Ticker(symbol_list[0]).history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                raise HTTPException(status_code=404, detail=f"No data found for {symbol_list[0]}")
            
            return {
                "symbol": symbol_list[0],
                "period": period,
                "interval": interval,
                "data": format_stock_data(data, {})
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing instruments: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while comparing instruments")

# ✅ 8️⃣ Get Popular Indian Mutual Funds by Category
@app.get("/mutual-funds/india/popular")
def get_popular_indian_mutual_funds(
    category: Optional[str] = Query(None, description="Fund category: equity, debt, hybrid, etc.")
):
    # This would ideally come from a database, but for demonstration:
    popular_funds = {
        "equity": [
            {"name": "HDFC Top 100 Fund", "symbol": "0P0000XVOI.BO"},
            {"name": "SBI Bluechip Fund", "symbol": "0P0000YCNI.BO"},
            {"name": "Axis Bluechip Fund", "symbol": "0P0000Z5X9.BO"},
            {"name": "Mirae Asset Large Cap Fund", "symbol": "0P0000YD2F.BO"}
        ],
        "debt": [
            {"name": "HDFC Corporate Bond Fund", "symbol": "0P0000YWLG.BO"},
            {"name": "SBI Corporate Bond Fund", "symbol": "0P0000ZM1O.BO"},
            {"name": "Kotak Corporate Bond Fund", "symbol": "0P0000Y5QE.BO"}
        ],
        "hybrid": [
            {"name": "ICICI Prudential Balanced Advantage Fund", "symbol": "0P0000XVE2.BO"},
            {"name": "HDFC Balanced Advantage Fund", "symbol": "0P0000XV7Y.BO"}
        ]
    }
    
    try:
        if category and category.lower() in popular_funds:
            return {"funds": popular_funds[category.lower()]}
        else:
            return {"funds": {cat: funds for cat, funds in popular_funds.items()}}
    except Exception as e:
        logger.error(f"Error fetching popular mutual funds: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while fetching popular mutual funds")

# Run FastAPI server
if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8080))  # Use environment PORT or default to 8080
    uvicorn.run(app, host="0.0.0.0", port=port) 