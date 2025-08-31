from langchain_core.tools import BaseTool
from models.database import DatabaseManager
from models.data_models import Portfolio
from datetime import datetime
import yfinance as yf
import logging
from typing import Optional

class AddPortfolioSharesTool(BaseTool):
    name: str = "add_portfolio_shares"
    description: str = "Add shares to user's portfolio"
    
    _db_manager: Optional[DatabaseManager] = None
    
    @classmethod
    def set_db_manager(cls, db_manager: DatabaseManager):
        cls._db_manager = db_manager
    
    def _run(self, user_id: str, symbol: str, quantity: float) -> str:
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        
        try:
            logger.debug(f"AddPortfolioSharesTool: user_id={user_id}, symbol={symbol}, quantity={quantity}")
            symbol_mapping = {
                'RELIANCE': 'RELIANCE.NS', 'TCS': 'TCS.NS', 'HDFC': 'HDFCBANK.NS',
                'HDFCBANK': 'HDFCBANK.NS', 'INFY': 'INFY.NS', 'INFOSYS': 'INFY.NS',
                'ICICIBANK': 'ICICIBANK.NS', 'ICICI': 'ICICIBANK.NS', 'SBIN': 'SBIN.NS',
                'SBI': 'SBIN.NS', 'ITC': 'ITC.NS', 'WIPRO': 'WIPRO.NS', 'LT': 'LT.NS',
                'LARSEN': 'LT.NS', 'HCLTECH': 'HCLTECH.NS', 'HCL': 'HCLTECH.NS',
                'BAJFINANCE': 'BAJFINANCE.NS', 'BAJAJ': 'BAJFINANCE.NS',
                'MARUTI': 'MARUTI.NS', 'ASIANPAINT': 'ASIANPAINT.NS', 'ASIAN': 'ASIANPAINT.NS'
            }
            symbol = symbol.upper().strip()
            logger.debug(f"Original symbol: {symbol}")
            if symbol in symbol_mapping:
                symbol = symbol_mapping[symbol]
            elif not symbol.endswith(('.NS', '.BO')):
                symbol = symbol + '.NS'
            logger.debug(f"Normalized symbol: {symbol}")
            
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1d")
            logger.debug(f"yfinance history for {symbol}: {'non-empty' if not hist.empty else 'empty'}")
            if hist.empty:
                if symbol.endswith('.NS'):
                    symbol = symbol.replace('.NS', '')
                    stock = yf.Ticker(symbol)
                    hist = stock.history(period="1d")
                    logger.debug(f"Retried yfinance history for {symbol}: {'non-empty' if not hist.empty else 'empty'}")
                if hist.empty:
                    return f"❌ Invalid stock symbol: {symbol}. Try RELIANCE, TCS, etc."
            
            portfolio = self._db_manager.get_portfolio(user_id)
            logger.debug(f"Portfolio retrieved: {portfolio}")
            if not portfolio:
                portfolio = Portfolio(
                    user_id=user_id,
                    holdings={},
                    total_value=0.0,
                    last_updated=datetime.now()
                )
                logger.debug("Created new portfolio")
            
            portfolio.holdings[symbol] = portfolio.holdings.get(symbol, 0) + quantity
            current_price = hist['Close'].iloc[-1]
            portfolio.total_value = sum(
                yf.Ticker(s).history(period="1d")['Close'].iloc[-1] * q
                for s, q in portfolio.holdings.items()
                if not yf.Ticker(s).history(period="1d").empty
            )
            portfolio.last_updated = datetime.now()
            logger.debug(f"Updated holdings: {portfolio.holdings}, total_value: {portfolio.total_value}")
            self._db_manager.save_portfolio(portfolio)
            logger.debug("Portfolio saved to DB")
            
            return f"✅ Added {quantity} shares of {symbol} to your portfolio. Current value: ₹{portfolio.total_value:.2f}"
        except Exception as e:
            logger.error(f"Error in AddPortfolioSharesTool: {str(e)}", exc_info=True)
            return f"❌ Error adding shares: {str(e)}. Please check the symbol or try again later."
    
    async def _arun(self, user_id: str, symbol: str, quantity: float) -> str:
        return self._run(user_id, symbol, quantity)