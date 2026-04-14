from mcp.server.fastmcp import FastMCP

# 1. Initialize the MCP Server (This is our "USB-C port")
mcp = FastMCP("leave_manager")

# 2. Mock Database for the HR System
# In a real app, this would connect to SQLite or a real HR API.
mock_db = {
    "E001": {
        "name": "John Doe",
        "balance": 18,
        "history": ["2025-12-25", "2026-01-01"]
    },
    "E002": {
        "name": "Jane Smith",
        "balance": 20,
        "history": []
    }
}

# 3. The Tools
# By adding @mcp.tool(), we tell the server to broadcast this function to any connected AI.
# The docstrings ("""...""") are CRITICAL. The AI reads them to know how and when to use the tool.

@mcp.tool()
def get_leave_balance(employee_id: str) -> str:
    """Get the available leave balance (in days) for a specific employee ID."""
    employee = mock_db.get(employee_id)
    if employee:
        return f"Employee {employee_id} ({employee['name']}) has {employee['balance']} days of leave remaining."
    return f"Error: Employee {employee_id} not found."

@mcp.tool()
def get_leave_history(employee_id: str) -> str:
    """Get the leave history (dates already taken) for a specific employee ID."""
    employee = mock_db.get(employee_id)
    if employee:
        history = employee['history']
        if not history:
            return f"Employee {employee_id} has not taken any leave yet."
        return f"Employee {employee_id} has taken leave on: {', '.join(history)}"
    return f"Error: Employee {employee_id} not found."

@mcp.tool()
def apply_leave(employee_id: str, leave_dates: list[str]) -> str:
    """Apply for a leave. The dates MUST be strictly formatted as YYYY-MM-DD strings in a list."""
    employee = mock_db.get(employee_id)
    if not employee:
        return f"Error: Employee {employee_id} not found."
    
    days_requested = len(leave_dates)
    if employee["balance"] >= days_requested:
        employee["balance"] -= days_requested
        employee["history"].extend(leave_dates)
        return f"Success! Applied for {days_requested} day(s) of leave for {employee_id}. New balance: {employee['balance']} days."
    else:
        return f"Error: Insufficient balance. {employee_id} only has {employee['balance']} days left."

# 4. Resources
# Resources are static data or greetings that the server can provide.
@mcp.resource("greeting://hr")
def hr_greeting() -> str:
    """A simple greeting resource from the HR system."""
    return "Hello! I am the HR Leave Manager Assistant. How can I help you today?"

# 5. Run the Server
if __name__ == "__main__":
    # When executed, this starts listening for MCP Client connections via standard input/output
    mcp.run()