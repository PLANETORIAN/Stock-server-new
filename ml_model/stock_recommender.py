import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import yfinance as yf
import logging
from typing import List, Dict, Any, Tuple, Optional
import joblib
import os
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockRecommender:
    """
    A recommendation system for stocks based on various financial metrics
    """
    
    def __init__(self, model_path: str = "ml_model/models"):
        """
        Initialize the recommender
        
        Args:
            model_path: Path to save/load models
        """
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.feature_columns = [
            'returns_mean', 'returns_std', 'beta', 'alpha', 
            'sharpe_ratio', 'max_drawdown', 'market_cap_log',
            'volume_mean', 'volatility'
        ]
        self.similarity_matrix = None
        self.stock_data = None
        self.stock_features = None
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
    def _fetch_stock_data(self, tickers: List[str], period: str = "1y") -> pd.DataFrame:
        """
        Fetch historical data for a list of tickers
        """
        logger.info(f"Fetching data for {len(tickers)} tickers")
        
        all_data = []
        for ticker in tickers:
            # Skip comments or empty lines
            if ticker.startswith('#') or not ticker.strip():
                continue
                
            try:
                # Ensure Indian stocks use .NS suffix, but skip for indices (starting with ^) and mutual funds
                if not ticker.startswith('^') and not ticker.endswith('.NS') and not ticker.endswith('.BO') and not ticker.startswith('0P'):
                    # Check if it's an Indian stock by trying with .NS suffix
                    try:
                        test_stock = yf.Ticker(f"{ticker}.NS")
                        test_info = test_stock.info
                        if test_info and test_info.get('exchange') == 'NSE':
                            ticker = f"{ticker}.NS"
                    except Exception:
                        # If there's an error, just use the original symbol (might be a mutual fund)
                        pass
                
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                
                if not hist.empty:
                    # Calculate returns
                    hist['Returns'] = hist['Close'].pct_change()
                    
                    # Calculate metrics
                    returns_mean = hist['Returns'].mean() * 252  # Annualized
                    returns_std = hist['Returns'].std() * np.sqrt(252)  # Annualized
                    volatility = returns_std
                    
                    # Get market data for beta calculation
                    market = yf.Ticker("^GSPC")
                    market_hist = market.history(period=period)
                    market_hist['Returns'] = market_hist['Close'].pct_change()
                    
                    # Align dates
                    aligned_data = pd.concat([
                        hist['Returns'].rename('stock_returns'),
                        market_hist['Returns'].rename('market_returns')
                    ], axis=1).dropna()
                    
                    # Calculate beta
                    covariance = aligned_data['stock_returns'].cov(aligned_data['market_returns'])
                    market_variance = aligned_data['market_returns'].var()
                    beta = covariance / market_variance if market_variance != 0 else 1.0
                    
                    # Calculate alpha
                    risk_free_rate = 0.04  # 4% annual
                    alpha = returns_mean - (risk_free_rate + beta * (market_hist['Returns'].mean() * 252 - risk_free_rate))
                    
                    # Calculate Sharpe ratio
                    excess_returns = hist['Returns'] - (1 + risk_free_rate) ** (1/252) - 1
                    sharpe = excess_returns.mean() * 252 / (excess_returns.std() * np.sqrt(252)) if excess_returns.std() != 0 else 0
                    
                    # Calculate maximum drawdown
                    rolling_max = hist['Close'].expanding().max()
                    drawdowns = hist['Close'] / rolling_max - 1
                    max_drawdown = drawdowns.min()
                    
                    # Get additional info
                    info = stock.info
                    market_cap = info.get('marketCap', 0)
                    market_cap_log = np.log1p(market_cap) if market_cap > 0 else 0
                    volume_mean = hist['Volume'].mean() if 'Volume' in hist.columns else 0
                    
                    # Create feature vector
                    stock_features = {
                        'ticker': ticker,
                        'name': info.get('shortName', ticker),
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown'),
                        'returns_mean': returns_mean,
                        'returns_std': returns_std,
                        'beta': beta,
                        'alpha': alpha,
                        'sharpe_ratio': sharpe,
                        'max_drawdown': max_drawdown,
                        'market_cap_log': market_cap_log,
                        'volume_mean': volume_mean,
                        'volatility': volatility
                    }
                    
                    all_data.append(stock_features)
                    logger.info(f"Processed {ticker}")
                else:
                    logger.warning(f"No data found for {ticker}")
            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
        
        return pd.DataFrame(all_data)
    
    def train(self, tickers: List[str], period: str = "1y"):
        """
        Train the recommendation model
        
        Args:
            tickers: List of stock tickers to train on
            period: Time period for historical data
        """
        logger.info(f"Training model with {len(tickers)} tickers")
        
        # Fetch data
        self.stock_data = self._fetch_stock_data(tickers, period)
        
        if self.stock_data.empty:
            logger.error("No data available for training")
            return False
        
        # Extract features
        self.stock_features = self.stock_data[self.feature_columns].copy()
        
        # Handle missing values with imputation
        self.stock_features_imputed = self.imputer.fit_transform(self.stock_features)
        
        # Scale features
        self.stock_features_scaled = self.scaler.fit_transform(self.stock_features_imputed)
        
        # Apply PCA for dimensionality reduction
        self.stock_features_pca = self.pca.fit_transform(self.stock_features_scaled)
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(self.stock_features_pca)
        
        # Save models
        self._save_models()
        
        logger.info("Model training completed")
        return True
    
    def _save_models(self):
        """Save trained models"""
        joblib.dump(self.scaler, os.path.join(self.model_path, "scaler.joblib"))
        joblib.dump(self.imputer, os.path.join(self.model_path, "imputer.joblib"))
        joblib.dump(self.pca, os.path.join(self.model_path, "pca.joblib"))
        joblib.dump(self.similarity_matrix, os.path.join(self.model_path, "similarity_matrix.joblib"))
        self.stock_data.to_csv(os.path.join(self.model_path, "stock_data.csv"), index=False)
        logger.info("Models saved successfully")
    
    def _load_models(self) -> bool:
        """Load trained models"""
        try:
            self.scaler = joblib.load(os.path.join(self.model_path, "scaler.joblib"))
            self.imputer = joblib.load(os.path.join(self.model_path, "imputer.joblib"))
            self.pca = joblib.load(os.path.join(self.model_path, "pca.joblib"))
            self.similarity_matrix = joblib.load(os.path.join(self.model_path, "similarity_matrix.joblib"))
            self.stock_data = pd.read_csv(os.path.join(self.model_path, "stock_data.csv"))
            logger.info("Models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def recommend(self, ticker: str, n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """
        Recommend similar stocks based on a given ticker
        
        Args:
            ticker: Stock ticker to base recommendations on
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended stocks with similarity scores
        """
        # Load models if not already loaded
        if self.similarity_matrix is None:
            if not self._load_models():
                logger.error("Models not loaded. Please train the model first.")
                return []
        
        # Find the index of the ticker
        try:
            ticker_idx = self.stock_data[self.stock_data['ticker'] == ticker].index[0]
        except IndexError:
            logger.error(f"Ticker {ticker} not found in trained data")
            return []
        
        # Get similarity scores
        similarity_scores = self.similarity_matrix[ticker_idx]
        
        # Get top N similar stocks (excluding the input ticker)
        similar_indices = np.argsort(similarity_scores)[::-1][1:n_recommendations+1]
        
        # Format recommendations
        recommendations = []
        for idx in similar_indices:
            stock = self.stock_data.iloc[idx]
            recommendations.append({
                'ticker': stock['ticker'],
                'name': stock['name'],
                'sector': stock['sector'],
                'industry': stock['industry'],
                'similarity_score': float(similarity_scores[idx]),
                'metrics': {
                    'returns_mean': float(stock['returns_mean']),
                    'beta': float(stock['beta']),
                    'alpha': float(stock['alpha']),
                    'sharpe_ratio': float(stock['sharpe_ratio']),
                    'max_drawdown': float(stock['max_drawdown'])
                }
            })
        
        return recommendations
    
    def recommend_by_metrics(self, 
                            target_metrics: Dict[str, float], 
                            n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """
        Recommend stocks based on target metrics
        
        Args:
            target_metrics: Dictionary of target metrics (e.g., {'beta': 1.2, 'sharpe_ratio': 1.5})
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended stocks with similarity scores
        """
        # Load models if not already loaded
        if self.stock_features is None:
            if not self._load_models():
                logger.error("Models not loaded. Please train the model first.")
                return []
        
        # Create a target feature vector
        target_vector = np.zeros(len(self.feature_columns))
        for i, col in enumerate(self.feature_columns):
            if col in target_metrics:
                target_vector[i] = target_metrics[col]
        
        # Scale the target vector
        target_vector_scaled = self.scaler.transform(target_vector.reshape(1, -1))
        
        # Apply PCA
        target_vector_pca = self.pca.transform(target_vector_scaled)
        
        # Calculate similarity to all stocks
        similarity_scores = cosine_similarity(target_vector_pca, self.stock_features_pca)[0]
        
        # Get top N similar stocks
        similar_indices = np.argsort(similarity_scores)[::-1][:n_recommendations]
        
        # Format recommendations
        recommendations = []
        for idx in similar_indices:
            stock = self.stock_data.iloc[idx]
            recommendations.append({
                'ticker': stock['ticker'],
                'name': stock['name'],
                'sector': stock['sector'],
                'industry': stock['industry'],
                'similarity_score': float(similarity_scores[idx]),
                'metrics': {
                    'returns_mean': float(stock['returns_mean']),
                    'beta': float(stock['beta']),
                    'alpha': float(stock['alpha']),
                    'sharpe_ratio': float(stock['sharpe_ratio']),
                    'max_drawdown': float(stock['max_drawdown'])
                }
            })
        
        return recommendations
    
    def get_sector_recommendations(self, sector: str, n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """
        Recommend top performing stocks in a specific sector
        
        Args:
            sector: Sector name
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended stocks with performance metrics
        """
        # Load models if not already loaded
        if self.stock_data is None:
            if not self._load_models():
                logger.error("Models not loaded. Please train the model first.")
                return []
        
        # Filter stocks by sector
        sector_stocks = self.stock_data[self.stock_data['sector'] == sector]
        
        if sector_stocks.empty:
            logger.warning(f"No stocks found in sector: {sector}")
            return []
        
        # Sort by Sharpe ratio (risk-adjusted return)
        top_stocks = sector_stocks.nlargest(n_recommendations, 'sharpe_ratio')
        
        # Format recommendations
        recommendations = []
        for _, stock in top_stocks.iterrows():
            recommendations.append({
                'ticker': stock['ticker'],
                'name': stock['name'],
                'sector': stock['sector'],
                'industry': stock['industry'],
                'metrics': {
                    'returns_mean': float(stock['returns_mean']),
                    'beta': float(stock['beta']),
                    'alpha': float(stock['alpha']),
                    'sharpe_ratio': float(stock['sharpe_ratio']),
                    'max_drawdown': float(stock['max_drawdown'])
                }
            })
        
        return recommendations 