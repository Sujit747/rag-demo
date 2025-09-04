import streamlit as st
import streamlit.components.v1 as components
import asyncio
import json
import os
import logging
from core.assistant import FinancialAssistant
import yfinance as yf
from tools.document_rag import DocumentRAGTool

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

    # Inject custom CSS for animations and logo-text styling
    st.markdown("""
        <style>
            /* Fade-in animation for title */
            .main-title {
                animation: fadeIn 1.5s ease-in-out;
            }
            @keyframes fadeIn {
                0% { opacity: 0; transform: translateY(-20px); }
                100% { opacity: 1; transform: translateY(0); }
            }

            /* Sidebar logo and text container */
            .logo-text-container {
                display: flex;
                align-items: center;
                margin-bottom: 20px;
            }
            .logo-text {
                font-size: 24px;
                font-weight: bold;
                color: #4CAF50;
                margin-left: 10px;
                animation: textPop 1s ease-in-out;
            }
            @keyframes textPop {
                0% { opacity: 0; transform: scale(0.8); }
                100% { opacity: 1; transform: scale(1); }
            }

            /* Sidebar elements slide-in animation */
            .sidebar .sidebar-content > div {
                animation: slideIn 0.5s ease-in-out;
            }
            @keyframes slideIn {
                0% { opacity: 0; transform: translateX(-20px); }
                100% { opacity: 1; transform: translateX(0); }
            }

            /* Chat message animation */
            .stChatMessage {
                animation: messagePop 0.5s ease-in-out;
            }
            @keyframes messagePop {
                0% { opacity: 0; transform: scale(0.9); }
                100% { opacity: 1; transform: scale(1); }
            }
        </style>
    """, unsafe_allow_html=True)

    # Display local JPEG logo and "Infinity Pool" text in sidebar
    with st.sidebar:
        st.markdown('<div class="logo-text-container">', unsafe_allow_html=True)
        st.image("images/logo.jpeg", width=90)  # Replace with actual path to your JPEG
        st.markdown('<span class="logo-text">Infinity Pool</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.header("⚙️ Configuration")
        user_id = st.text_input(
            "User ID",
            value="demo_user_123",
            help="Unique identifier for user sessions"
        )
        
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
        
        st.markdown("### 📄 Document Upload")
        uploaded_file = st.file_uploader("Upload a PDF for RAG", type=["pdf"])
        rag_tool = DocumentRAGTool()
        if uploaded_file:
            success, message = rag_tool.handle_file_upload(uploaded_file)
            st.text(message)

    st.markdown('<h1 class="main-title">🤖 Financial Assistant Demo - FAISS Edition</h1>', unsafe_allow_html=True)
    st.subheader("Powered by LangChain + LangGraph + LlamaIndex + FAISS")

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

    if prompt := st.chat_input("Ask me about stocks, portfolio, SIPs, or document content..."):
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

if __name__ == "__main__":
    main()