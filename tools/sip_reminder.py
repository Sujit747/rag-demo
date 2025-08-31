from langchain_core.tools import BaseTool
from models.database import DatabaseManager
from models.data_models import FinancialGoal
from datetime import datetime
from typing import Optional

class SIPReminderTool(BaseTool):
    name: str = "sip_reminder_check"
    description: str = "Check for upcoming SIP payments"
    
    _db_manager: Optional[DatabaseManager] = None
    
    @classmethod
    def set_db_manager(cls, db_manager: DatabaseManager):
        cls._db_manager = db_manager
    
    def _run(self, user_id: str) -> str:
        try:
            goals = self._db_manager.get_financial_goals(user_id)
            sip_data = [
                {'fund_name': goal.goal_type, 'amount': goal.target_amount, 'due_date': goal.deadline.strftime('%dth every month')}
                for goal in goals if goal.goal_type == 'SIP'
            ]
            if not sip_data:
                return "No SIP goals set up. Consider adding some!"
            
            today = datetime.now()
            reminders = []
            
            for sip in sip_data:
                if '15th' in sip['due_date'] and today.day >= 13:
                    reminders.append(f"🔔 SIP Due: {sip['fund_name']} - ₹{sip['amount']} on 15th")
                elif '10th' in sip['due_date'] and today.day >= 8:
                    reminders.append(f"🔔 SIP Due: {sip['fund_name']} - ₹{sip['amount']} on 10th")
                elif '25th' in sip['due_date'] and today.day >= 23:
                    reminders.append(f"🔔 SIP Due: {sip['fund_name']} - ₹{sip['amount']} on 25th")
            
            if reminders:
                return "Your SIP Reminders:\n" + "\n".join(reminders)
            else:
                return "No SIP payments due in the next few days. All good! 👍"
                
        except Exception as e:
            return f"Error checking SIP reminders: {str(e)}"
    
    async def _arun(self, user_id: str) -> str:
        return self._run(user_id)