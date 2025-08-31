from langchain_core.tools import BaseTool
import yfinance as yf
from datetime import datetime

class StockPriceTool(BaseTool):
    name: str = "stock_price_lookup"
    description: str = "Get current stock price and basic info for Indian/US stocks"
    
    def _run(self, symbol: str) -> str:
        try:
            symbol_mapping = {
                'RELIANCE': 'RELIANCE.NS', 'TCS': 'TCS.NS', 'HDFC': 'HDFCBANK.NS',
                'HDFCBANK': 'HDFCBANK.NS', 'INFY': 'INFY.NS', 'INFOSYS': 'INFY.NS',
                'ICICIBANK': 'ICICIBANK.NS', 'ICICI': 'ICICIBANK.NS', 'SBIN': 'SBIN.NS',
                'SBI': 'SBIN.NS', 'ITC': 'ITC.NS', 'WIPRO': 'WIPRO.NS', 'LT': 'LT.NS',
                'LARSEN': 'LT.NS', 'HCLTECH': 'HCLTECH.NS', 'HCL': 'HCLTECH.NS',
                'BAJFINANCE': 'BAJFINANCE.NS', 'BAJAJ': 'BAJFINANCE.NS',
                'MARUTI': 'MARUTI.NS', 'ASIANPAINT': 'ASIANPAINT.NS', 'ASIAN': 'ASIANPAINT.NS'
            }
            
            original_symbol = symbol.upper().strip()
            if original_symbol in symbol_mapping:
                symbol = symbol_mapping[original_symbol]
            elif original_symbol.endswith(('.NS', '.BO')):
                symbol = original_symbol
            elif len(original_symbol) <= 10 and original_symbol.isalpha():
                symbol = original_symbol + '.NS'
            else:
                symbol = original_symbol
            
            print(f"Looking up symbol: {symbol}")
            
            stock = yf.Ticker(symbol)
            hist = None
            for period in ["1d", "5d", "1mo"]:
                try:
                    hist = stock.history(period=period)
                    if not hist.empty:
                        break
                except:
                    continue
            
            if hist is None or hist.empty:
                if symbol.endswith('.NS'):
                    symbol_without_ns = symbol.replace('.NS', '')
                    stock = yf.Ticker(symbol_without_ns)
                    hist = stock.history(period="1d")
                
                if hist is None or hist.empty:
                    return f"❌ Could not find data for {original_symbol}. Please check the symbol name."
            
            try:
                info = stock.info
            except:
                info = {}
            
            current_price = hist['Close'].iloc[-1]
            previous_close = hist['Open'].iloc[0] if len(hist) > 1 else current_price
            change = current_price - previous_close
            change_pct = (change / previous_close) * 100 if previous_close != 0 else 0
            currency = "₹" if symbol.endswith(('.NS', '.BO')) else "$"
            
            return f"""
📈 **{info.get('longName', symbol)} ({symbol})**

💰 **Current Price**: {currency}{current_price:.2f}
📊 **Change**: {currency}{change:.2f} ({change_pct:+.2f}%)
🏢 **Company**: {info.get('shortName', info.get('longName', 'N/A'))}
🏭 **Sector**: {info.get('sector', 'N/A')}
📅 **Last Updated**: {hist.index[-1].strftime('%Y-%m-%d %H:%M')}

💡 **Tip**: Stock data from Yahoo Finance
            """
        except Exception as e:
            return f"❌ Error fetching data for {symbol}: {str(e)}\n\n💡 Try using common names like: RELIANCE, TCS, HDFC, INFY, SBI"
    
    async def _arun(self, symbol: str) -> str:
        return self._run(symbol)