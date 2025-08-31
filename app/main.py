import streamlit as st
import streamlit.components.v1 as components
import asyncio
import json
import os
import logging
from core.assistant import FinancialAssistant
import yfinance as yf

def display_portfolio_chart(holdings, values):
    chart_config = {
        "type": "bar",
        "data": {
            "labels": list(holdings.keys()),
            "datasets": [{
                "label": "Portfolio Holdings Value (₹)",
                "data": values,
                "backgroundColor": ["#4CAF50", "#2196F3", "#FFC107", "#FF5722"],
                "borderColor": ["#388E3C", "#1976D2", "#FFA000", "#D81B60"],
                "borderWidth": 1
            }]
        },
        "options": {
            "scales": {
                "y": {"beginAtZero": True, "title": {"display": True, "text": "Value (₹)"}},
                "x": {"title": {"display": True, "text": "Stocks"}}
            },
            "plugins": {"title": {"display": True, "text": "Portfolio Holdings"}}
        }
    }
    components.html(f"""
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <canvas id="portfolioChart" width="400" height="200"></canvas>
        <script>
            const ctx = document.getElementById('portfolioChart').getContext('2d');
            new Chart(ctx, {json.dumps(chart_config)});
        </script>
    """, height=300)

def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('financial_assistant.log')
        ]
    )
    logger = logging.getLogger(__name__)

    st.set_page_config(
        page_title="Financial Assistant Demo - FAISS Edition",
        page_icon="💰",
        layout="wide"
    )
    st.title("🤖 Financial Assistant Demo - FAISS Edition")
    st.subheader("Powered by LangChain + LangGraph + LlamaIndex + FAISS")

    with st.sidebar:
        st.header("⚙️ Configuration")
        user_id = st.text_input(
            "User ID",
            value="demo_user_123",
            help="Unique identifier for user sessions"
        )
        st.markdown("---")
        st.markdown("### 🔧 Framework Stack")
        st.markdown("- **LangChain**: Agent & tool orchestration")
        st.markdown("- **LangGraph**: Multi-step workflow management")
        st.markdown("- **LlamaIndex**: Context & memory management")
        st.markdown("- **FAISS**: High-performance vector similarity search")
        st.markdown("- **yFinance**: Real financial data")
        st.markdown("---")
        st.markdown("### 🎯 Demo Features")
        st.markdown("- Stock price lookup")
        st.markdown("- Portfolio analysis")
        st.markdown("- SIP reminders")
        st.markdown("- Add portfolio shares")
        st.markdown("- Set SIP reminders")
        st.markdown("- Long-term memory with FAISS")
        st.markdown("---")
        
        if 'assistant' in st.session_state:
            st.markdown("### 📊 FAISS Memory Stats")
            try:
                stats = st.session_state.assistant.get_memory_stats()
                st.metric("Total Vectors", stats["total_vectors"])
                st.metric("Metadata Entries", stats["total_metadata"])
                st.metric("Vector Dimension", stats["dimension"])
                st.text(f"Index Type: {stats['index_type']}")
            except Exception as e:
                st.error(f"Error loading stats: {str(e)}")
        
        st.markdown("---")
        st.markdown("### 📜 Debug Log")
        if os.path.exists('financial_assistant.log'):
            with open('financial_assistant.log', 'r') as f:
                st.text_area("Recent Logs", f.read(), height=200)

    if 'assistant' not in st.session_state:
        try:
            logger.info("Initializing Financial Assistant with FAISS")
            with st.spinner("Initializing FAISS vector database..."):
                st.session_state.assistant = FinancialAssistant()
            st.success("✅ Financial Assistant with FAISS ready!")
        except Exception as e:
            logger.error(f"Error initializing assistant: {str(e)}", exc_info=True)
            st.error(f"❌ Error initializing assistant: {str(e)}")
            return

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask me about stocks, portfolio, SIPs, or financial advice..."):
        logger.info(f"Processing user prompt: {prompt}")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Processing your request with FAISS memory..."):
                try:
                    response = asyncio.run(
                        st.session_state.assistant.process_message(user_id, prompt)
                    )
                    logger.debug(f"Assistant response: {response}")
                    st.write(response)
                    if "portfolio" in response.lower() or "added" in response.lower():
                        portfolio = st.session_state.assistant.db_manager.get_portfolio(user_id)
                        if portfolio:
                            logger.debug(f"Portfolio for chart: {portfolio.holdings}")
                            values = []
                            valid_holdings = {}
                            for symbol, quantity in portfolio.holdings.items():
                                try:
                                    stock = yf.Ticker(symbol)
                                    hist = stock.history(period="1d")
                                    if not hist.empty:
                                        value = hist['Close'].iloc[-1] * quantity
                                        values.append(value)
                                        valid_holdings[symbol] = quantity
                                        logger.debug(f"Chart data for {symbol}: value={value}")
                                    else:
                                        values.append(0.0)
                                        logger.warning(f"No data for {symbol}")
                                        st.warning(f"No data for {symbol}")
                                except Exception as e:
                                    values.append(0.0)
                                    logger.error(f"Error fetching data for {symbol}: {str(e)}")
                                    st.warning(f"Error fetching data for {symbol}: {str(e)}")
                            if any(v > 0 for v in values):
                                display_portfolio_chart(valid_holdings, values)
                            else:
                                logger.error("No valid stock data available for chart")
                                st.error("No valid stock data available for chart.")
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    logger.error(f"Error processing prompt: {error_msg}", exc_info=True)
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    st.markdown("---")
    st.markdown("### 💡 Try These Examples:")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("📈 Get RELIANCE price"):
            example_query = "What's the current price of RELIANCE stock?"
            st.session_state.messages.append({"role": "user", "content": example_query})
            st.rerun()
    with col2:
        if st.button("📊 Analyze portfolio"):
            example_query = "Can you analyze my portfolio performance?"
            st.session_state.messages.append({"role": "user", "content": example_query})
            st.rerun()
    with col3:
        if st.button("🔔 Check SIPs"):
            example_query = "Do I have any SIP payments due?"
            st.session_state.messages.append({"role": "user", "content": example_query})
            st.rerun()
    with col4:
        if st.button("➕ Add shares"):
            example_query = "Add 10 RELIANCE shares to my portfolio"
            st.session_state.messages.append({"role": "user", "content": example_query})
            st.rerun()
    with col5:
        if st.button("⏰ Set SIP"):
            example_query = "Set ₹5000 SIP for HDFC Fund on 15th"
            st.session_state.messages.append({"role": "user", "content": example_query})
            st.rerun()

    st.markdown("---")
    st.markdown("### 🧠 FAISS Memory Features:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Query Memory"):
            example_query = "What did we discuss about my investment preferences?"
            st.session_state.messages.append({"role": "user", "content": example_query})
            st.rerun()
    with col2:
        if st.button("📈 Memory Context"):
            if 'assistant' in st.session_state:
                context = st.session_state.assistant.get_user_insights(user_id)
                st.info(f"User Context:\n{context}")
            else:
                st.warning("Assistant not initialized yet")

if __name__ == "__main__":
    main()