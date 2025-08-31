import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from models.database import DatabaseManager
from memory.faiss_memory import LlamaIndexMemoryManagerFAISS
from tools.stock_price import StockPriceTool
from tools.portfolio import PortfolioAnalysisTool
from tools.sip_reminder import SIPReminderTool
from tools.add_shares import AddPortfolioSharesTool
from tools.set_sip import SetSIPReminderTool
from workflow.graph import FinancialWorkflowGraph

load_dotenv()

class FinancialAssistant:
    def __init__(self):
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        self.db_manager = DatabaseManager()
        self.memory_manager = LlamaIndexMemoryManagerFAISS(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-3.5-turbo",
            api_key=openai_api_key
        )
        
        portfolio_tool = PortfolioAnalysisTool()
        PortfolioAnalysisTool.set_db_manager(self.db_manager)
        sip_tool = SIPReminderTool()
        SIPReminderTool.set_db_manager(self.db_manager)
        add_shares_tool = AddPortfolioSharesTool()
        AddPortfolioSharesTool.set_db_manager(self.db_manager)
        set_sip_tool = SetSIPReminderTool()
        SetSIPReminderTool.set_db_manager(self.db_manager)
        
        self.tools = [
            StockPriceTool(),
            portfolio_tool,
            sip_tool,
            add_shares_tool,
            set_sip_tool
        ]
        
        self.workflow_graph = FinancialWorkflowGraph(self.tools, self.llm, self.db_manager)
        print("✅ Financial Assistant initialized with LangChain + LangGraph + LlamaIndex + FAISS")
    
    async def process_message(self, user_id: str, message: str) -> str:
        initial_state = {
            "messages": [message],
            "user_id": user_id,
            "current_task": "",
            "context": {},
            "tool_results": {},
            "final_response": ""
        }
        try:
            final_state = await self.workflow_graph.app.ainvoke(initial_state)
            response = final_state["final_response"]
            self.memory_manager.add_interaction(user_id, message, response)
            return response
        except Exception as e:
            error_response = f"I encountered an error: {str(e)}. Please try again."
            self.memory_manager.add_interaction(user_id, message, error_response)
            return error_response
    
    def get_user_insights(self, user_id: str) -> str:
        return self.memory_manager.get_user_context(user_id)
    
    def get_memory_stats(self) -> dict:
        return self.memory_manager.get_stats()