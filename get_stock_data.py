import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
from ml_model.stock_recommender import StockRecommender

def calculate_beta(stock_returns, market_returns):
    """Calculate beta using covariance of stock returns with market returns"""
    covariance = stock_returns.cov(market_returns)
    market_variance = market_returns.var()
    return float(covariance / market_variance) if market_variance != 0 else 1.0

def calculate_alpha(stock_returns, market_returns, risk_free_rate=0.04):
    """Calculate alpha (excess return) using the CAPM model"""
    # Convert annual risk-free rate to daily
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    
    # Calculate average returns
    avg_stock_return = stock_returns.mean() * 252  # Annualized
    avg_market_return = market_returns.mean() * 252  # Annualized
    
    # Calculate beta
    beta = calculate_beta(stock_returns, market_returns)
    if beta is None:
        return None
        
    # Calculate alpha
    alpha = avg_stock_return - (risk_free_rate + beta * (avg_market_return - risk_free_rate))
    return float(alpha)

def calculate_max_drawdown(prices):
    """Calculate maximum drawdown from peak to trough"""
    # Calculate cumulative maximum
    rolling_max = prices.expanding().max()
    # Calculate drawdowns
    drawdowns = prices / rolling_max - 1
    # Get maximum drawdown
    max_drawdown = drawdowns.min()
    return float(max_drawdown) if not pd.isna(max_drawdown) else None

def calculate_sharpe_ratio(returns, risk_free_rate=0.04):
    """Calculate Sharpe ratio using daily returns"""
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

def calculate_trust_score(info, std_dev, max_dd, beta, sharpe):
    """Calculate trust score (1-5) based on various metrics"""
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

def determine_safety_level(trust_score):
    """Determine safety level based on trust score"""
    if trust_score >= 4.0:
        return "High"
    elif trust_score >= 2.5:
        return "Mid"
    else:
        return "Low"

def calculate_annual_return(data):
    """Calculate annual return from historical data"""
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

def get_stock_data(symbol, period="1y", interval="1d"):
    """Get stock data and calculate metrics"""
    try:
        # Ensure Indian stocks use .NS suffix, but skip for indices (starting with ^) and mutual funds
        if not symbol.startswith('^') and not symbol.endswith('.NS') and not symbol.endswith('.BO') and not symbol.startswith('0P'):
            # Check if it's an Indian stock by trying with .NS suffix
            try:
                test_stock = yf.Ticker(f"{symbol}.NS")
                test_info = test_stock.info
                if test_info and test_info.get('exchange') == 'NSE':
                    symbol = f"{symbol}.NS"
            except Exception:
                # If there's an error, just use the original symbol (might be a mutual fund)
                pass
        
        # Get stock data
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        
        if data.empty:
            print(f"No data found for {symbol}")
            return None
            
        # Ensure data index is timezone-naive
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
            
        # Calculate returns
        data['Returns'] = data['Close'].pct_change()
        
        # Calculate one-year standard deviation
        one_year_ago = datetime.now().replace(tzinfo=None) - timedelta(days=365)
        one_year_data = data[data.index >= one_year_ago]
        std_dev = one_year_data['Returns'].std()
        
        # Calculate maximum drawdown
        max_dd = calculate_max_drawdown(data['Close'])
        
        # Calculate Sharpe ratio
        sharpe = calculate_sharpe_ratio(one_year_data['Returns'])
        
        # Get stock info
        info = stock.info
        
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
            
            # Calculate beta using the same method as the recommender
            covariance = aligned_data['stock_returns'].cov(aligned_data['market_returns'])
            market_variance = aligned_data['market_returns'].var()
            beta = covariance / market_variance if market_variance != 0 else 1.0
            
            alpha = calculate_alpha(
                aligned_data['stock_returns'],
                aligned_data['market_returns']
            )
        except Exception as e:
            print(f"Error calculating beta/alpha: {str(e)}")
            beta = 1.0  # Use fallback value instead of None
            alpha = None
        
        # Calculate trust score and safety level
        trust_score = calculate_trust_score(info, std_dev, max_dd, beta, sharpe)
        safety_level = determine_safety_level(trust_score)
        
        # Calculate annual return
        annual_return = calculate_annual_return(data)
        
        # Format response
        result = {
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
            "data": {
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
        }
        
        return result
    except Exception as e:
        print(f"Error fetching stock data: {str(e)}")
        return None

def get_recommendations(symbol, n_recommendations=5):
    """Get stock recommendations using the StockRecommender class"""
    try:
        # Initialize the recommender
        recommender = StockRecommender()
        
        # Load models
        if not recommender._load_models():
            print("Error loading models. Please train the model first.")
            return None
            
        # Get recommendations
        recommendations = recommender.recommend(symbol, n_recommendations)
        
        return recommendations
    except Exception as e:
        print(f"Error getting recommendations: {str(e)}")
        return None

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python get_stock_data.py <symbol> [n_recommendations]")
        return
        
    symbol = sys.argv[1]
    n_recommendations = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    # Get stock data
    stock_data = get_stock_data(symbol)
    if stock_data:
        print("\nStock Data:")
        print(json.dumps(stock_data, indent=2))
        
    # Get recommendations
    recommendations = get_recommendations(symbol, n_recommendations)
    if recommendations:
        print("\nRecommendations:")
        print(json.dumps(recommendations, indent=2))

if __name__ == "__main__":
    main() 