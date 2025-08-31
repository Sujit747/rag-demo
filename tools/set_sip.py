from langchain_core.tools import BaseTool
from models.database import DatabaseManager
from models.data_models import FinancialGoal
from datetime import datetime
from typing import Optional

class SetSIPReminderTool(BaseTool):
    name: str = "set_sip_reminder"
    description: str = "Set a new SIP reminder for the user"
    
    _db_manager: Optional[DatabaseManager] = None
    
    @classmethod
    def set_db_manager(cls, db_manager: DatabaseManager):
        cls._db_manager = db_manager
    
    def _run(self, user_id: str, fund_name: str, amount: float, due_day: int) -> str:
        try:
            if not (1 <= due_day <= 31):
                return "❌ Due day must be between 1 and 31."
            if amount <= 0:
                return "❌ Amount must be positive."
            
            goal_id = f"{user_id}_sip_{fund_name}_{datetime.now().isoformat()}"
            goal = FinancialGoal(
                goal_id=goal_id,
                user_id=user_id,
                goal_type="SIP",
                target_amount=amount,
                current_amount=0.0,
                deadline=datetime.now().replace(day=due_day),
                status="Active"
            )
            self._db_manager.save_financial_goals([goal])
            return f"✅ Set SIP reminder: ₹{amount} for {fund_name} on day {due_day} of each month."
        except Exception as e:
            return f"❌ Error setting SIP reminder: {str(e)}"
    
    async def _arun(self, user_id: str, fund_name: str, amount: float, due_day: int) -> str:
        return self._run(user_id, fund_name, amount, due_day)