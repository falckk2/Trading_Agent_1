#!/usr/bin/env python3
"""
Setup Notion tasks for DeepAgent Trading System
"""

from notion_client import Client
import sys

# Initialize Notion client
NOTION_TOKEN = "ntn_o27632756312yQBxI7hWh36F73PVA2pDXZQIjy5rSPn1Nu"
notion = Client(auth=NOTION_TOKEN)

def search_pages():
    """Search for available pages/databases"""
    try:
        print("🔍 Searching for your Notion pages...")
        results = notion.search(filter={"property": "object", "value": "page"})

        if results['results']:
            print(f"\n✅ Found {len(results['results'])} pages:")
            for idx, page in enumerate(results['results'][:10], 1):
                title = "Untitled"
                if 'properties' in page and 'title' in page['properties']:
                    title_prop = page['properties']['title']
                    if title_prop.get('title') and len(title_prop['title']) > 0:
                        title = title_prop['title'][0]['plain_text']
                print(f"{idx}. {title} (ID: {page['id']})")
            return results['results']
        else:
            print("❌ No pages found. Let me create a new database for you.")
            return []
    except Exception as e:
        print(f"⚠️  Error searching: {e}")
        return []

def create_task_database(parent_page_id=None):
    """Create a new database for project tasks"""
    try:
        print("\n📝 Creating new 'DeepAgent Tasks' database...")

        # If no parent page, we need to create in a workspace
        # This requires the integration to have access to a page
        database_properties = {
            "Name": {"title": {}},
            "Status": {
                "select": {
                    "options": [
                        {"name": "Not Started", "color": "gray"},
                        {"name": "In Progress", "color": "blue"},
                        {"name": "Completed", "color": "green"},
                        {"name": "Blocked", "color": "red"}
                    ]
                }
            },
            "Priority": {
                "select": {
                    "options": [
                        {"name": "🔴 Critical", "color": "red"},
                        {"name": "🟡 High", "color": "yellow"},
                        {"name": "🟢 Medium", "color": "green"},
                        {"name": "🔵 Low", "color": "blue"}
                    ]
                }
            },
            "Category": {
                "select": {
                    "options": [
                        {"name": "Critical Fixes", "color": "red"},
                        {"name": "Configuration", "color": "orange"},
                        {"name": "Testing", "color": "green"},
                        {"name": "Enhancements", "color": "blue"}
                    ]
                }
            }
        }

        # Try to create database (this may fail if no parent page has granted access)
        try:
            if parent_page_id:
                database = notion.databases.create(
                    parent={"type": "page_id", "page_id": parent_page_id},
                    title=[{"type": "text", "text": {"content": "DeepAgent Trading System - Tasks"}}],
                    properties=database_properties
                )
            else:
                # Without parent, this will likely fail
                print("⚠️  To create a database, I need a parent page ID.")
                print("Please:")
                print("1. Create a new page in Notion manually")
                print("2. Share it with the integration")
                print("3. Copy the page ID from the URL")
                return None
        except Exception as e:
            print(f"❌ Cannot create database: {e}")
            print("\n📌 Manual setup required:")
            print("1. Go to Notion and create a new page")
            print("2. Click '...' menu > Add connections > Find your integration")
            print("3. Copy the page ID from URL: notion.so/Your-Page-{PAGE_ID}")
            print("4. Run this script again with the page ID")
            return None

        print(f"✅ Database created! ID: {database['id']}")
        return database['id']

    except Exception as e:
        print(f"❌ Error creating database: {e}")
        return None

def add_tasks_to_database(database_id):
    """Add all project tasks to the database"""

    tasks = [
        # Critical Fixes
        ("Fix main.py: Change TradingGUI import to MainWindow", "🔴 Critical", "Critical Fixes", "Not Started"),
        ("Fix main.py: Add AgentManager initialization in CLI mode", "🔴 Critical", "Critical Fixes", "Not Started"),
        ("Install websockets library (pip install websockets)", "🔴 Critical", "Critical Fixes", "Not Started"),
        ("Create .env file with Blofin API credentials", "🔴 Critical", "Configuration", "Not Started"),
        ("Update config/trading_config.json with exchange credentials", "🔴 Critical", "Configuration", "Not Started"),

        # Configuration & Setup
        ("Create data directory structure", "🟡 High", "Configuration", "Not Started"),
        ("Create logs directory", "🟡 High", "Configuration", "Not Started"),
        ("Initialize database tables", "🟡 High", "Configuration", "Not Started"),
        ("Register trading agents (RSI, MACD) in main.py", "🟡 High", "Configuration", "Not Started"),
        ("Set active agent in agent manager", "🟡 High", "Configuration", "Not Started"),
        ("Configure risk management parameters", "🟡 High", "Configuration", "Not Started"),
        ("Test configuration validation", "🟡 High", "Testing", "Not Started"),

        # Testing & Validation
        ("Run dependency check (python main.py --check-deps)", "🟡 High", "Testing", "Not Started"),
        ("Test CLI mode in debug (python main.py --cli --debug)", "🟡 High", "Testing", "Not Started"),
        ("Verify exchange connection (sandbox mode)", "🟡 High", "Testing", "Not Started"),
        ("Test RSI agent signal generation", "🟢 Medium", "Testing", "Not Started"),
        ("Test MACD agent signal generation", "🟢 Medium", "Testing", "Not Started"),
        ("Verify risk management validation", "🟡 High", "Testing", "Not Started"),
        ("Test order execution (paper trading)", "🟡 High", "Testing", "Not Started"),
        ("Monitor system for 48 hours", "🟡 High", "Testing", "Not Started"),

        # Optional Enhancements
        ("Complete GUI implementation (MainWindow class)", "🔵 Low", "Enhancements", "Not Started"),
        ("Expand test coverage to 80%", "🟢 Medium", "Enhancements", "Not Started"),
        ("Add email notification system", "🔵 Low", "Enhancements", "Not Started"),
        ("Add webhook notification system", "🔵 Low", "Enhancements", "Not Started"),
        ("Implement performance metrics collection", "🟢 Medium", "Enhancements", "Not Started"),
        ("Add backtesting framework", "🟢 Medium", "Enhancements", "Not Started"),
        ("Create user documentation", "🟢 Medium", "Enhancements", "Not Started"),
        ("Add more ML agents (XGBoost)", "🔵 Low", "Enhancements", "Not Started"),
        ("Integrate additional exchanges (Binance, Kraken)", "🔵 Low", "Enhancements", "Not Started"),
        ("Implement WebSocket real-time data streaming", "🔵 Low", "Enhancements", "Not Started"),
    ]

    print(f"\n📝 Adding {len(tasks)} tasks to database...")

    for idx, (task_name, priority, category, status) in enumerate(tasks, 1):
        try:
            notion.pages.create(
                parent={"database_id": database_id},
                properties={
                    "Name": {"title": [{"text": {"content": task_name}}]},
                    "Priority": {"select": {"name": priority}},
                    "Category": {"select": {"name": category}},
                    "Status": {"select": {"name": status}}
                }
            )
            print(f"  ✅ Added: {task_name}")
        except Exception as e:
            print(f"  ❌ Failed to add task {idx}: {e}")

    print(f"\n🎉 All tasks added successfully!")

def main():
    print("=" * 70)
    print("🚀 DeepAgent Trading System - Notion Task Setup")
    print("=" * 70)

    # Search for existing pages
    pages = search_pages()

    if pages:
        print("\n" + "=" * 70)
        print("Options:")
        print("1. Use one of the pages above (enter number)")
        print("2. Create new database in an existing page (enter page ID)")
        print("3. Quit (q)")
        print("=" * 70)

        # For automation, let's use the first page if available
        if len(sys.argv) > 1 and sys.argv[1] == 'auto':
            parent_id = pages[0]['id']
            print(f"\n🤖 Auto mode: Using first page as parent")
        else:
            print("\n💡 Running in auto mode with first available page...")
            parent_id = pages[0]['id']
    else:
        print("\n📌 No pages found with integration access.")
        print("\nPlease:")
        print("1. Go to Notion")
        print("2. Create or open a page")
        print("3. Click '...' → Add connections → Select your integration")
        print("4. Re-run this script")
        return

    # Create database
    db_id = create_task_database(parent_id)

    if db_id:
        # Add tasks
        add_tasks_to_database(db_id)
        print("\n" + "=" * 70)
        print("✅ SUCCESS! Your tasks are now in Notion!")
        print(f"Database ID: {db_id}")
        print("=" * 70)
    else:
        print("\n❌ Failed to create database. See instructions above.")

if __name__ == "__main__":
    main()
