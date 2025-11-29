#!/usr/bin/env python3
"""
Add tasks to existing Notion database
"""

from notion_client import Client

# Initialize Notion client
NOTION_TOKEN = "ntn_o27632756312yQBxI7hWh36F73PVA2pDXZQIjy5rSPn1Nu"
notion = Client(auth=NOTION_TOKEN)

# Use the database ID that was just created
DATABASE_ID = "da331291-af67-42ab-96b5-06505b7616d3"

def add_tasks():
    """Add all project tasks as simple pages"""

    tasks = [
        # Critical Fixes (Priority 1)
        ("🔴 Fix main.py: Change TradingGUI import to MainWindow", "Critical Fixes"),
        ("🔴 Fix main.py: Add AgentManager initialization in CLI mode", "Critical Fixes"),
        ("🔴 Install websockets library (pip install websockets)", "Critical Fixes"),
        ("🔴 Create .env file with Blofin API credentials", "Critical Fixes"),
        ("🔴 Update config/trading_config.json with exchange credentials", "Critical Fixes"),

        # Configuration & Setup (Priority 2)
        ("🟡 Create data directory structure", "Configuration"),
        ("🟡 Create logs directory", "Configuration"),
        ("🟡 Initialize database tables", "Configuration"),
        ("🟡 Register trading agents (RSI, MACD) in main.py", "Configuration"),
        ("🟡 Set active agent in agent manager", "Configuration"),
        ("🟡 Configure risk management parameters", "Configuration"),
        ("🟡 Test configuration validation", "Configuration"),

        # Testing & Validation (Priority 2-3)
        ("🟡 Run dependency check (python main.py --check-deps)", "Testing"),
        ("🟡 Test CLI mode in debug (python main.py --cli --debug)", "Testing"),
        ("🟡 Verify exchange connection (sandbox mode)", "Testing"),
        ("🟢 Test RSI agent signal generation", "Testing"),
        ("🟢 Test MACD agent signal generation", "Testing"),
        ("🟡 Verify risk management validation", "Testing"),
        ("🟡 Test order execution (paper trading)", "Testing"),
        ("🟡 Monitor system for 48 hours", "Testing"),

        # Optional Enhancements (Priority 4)
        ("🔵 Complete GUI implementation (MainWindow class)", "Enhancements"),
        ("🟢 Expand test coverage to 80%", "Enhancements"),
        ("🔵 Add email notification system", "Enhancements"),
        ("🔵 Add webhook notification system", "Enhancements"),
        ("🟢 Implement performance metrics collection", "Enhancements"),
        ("🟢 Add backtesting framework", "Enhancements"),
        ("🟢 Create user documentation", "Enhancements"),
        ("🔵 Add more ML agents (XGBoost)", "Enhancements"),
        ("🔵 Integrate additional exchanges (Binance, Kraken)", "Enhancements"),
        ("🔵 Implement WebSocket real-time data streaming", "Enhancements"),
    ]

    print(f"📝 Adding {len(tasks)} tasks to database...")
    print(f"Database ID: {DATABASE_ID}\n")

    success_count = 0
    for idx, (task_name, category) in enumerate(tasks, 1):
        try:
            # Simple approach - just add title
            notion.pages.create(
                parent={"database_id": DATABASE_ID},
                properties={
                    "Name": {
                        "title": [
                            {
                                "text": {
                                    "content": task_name
                                }
                            }
                        ]
                    }
                }
            )
            print(f"  ✅ {idx:2d}. {task_name}")
            success_count += 1
        except Exception as e:
            print(f"  ❌ {idx:2d}. Failed: {e}")

    print(f"\n{'='*70}")
    print(f"✅ Added {success_count}/{len(tasks)} tasks successfully!")
    print(f"{'='*70}")
    print(f"\n🔗 View in Notion:")
    print(f"https://notion.so/{DATABASE_ID.replace('-', '')}")

if __name__ == "__main__":
    add_tasks()
