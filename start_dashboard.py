"""
STRAT Trading Bot Dashboard Launcher
Choose between basic and advanced dashboard interfaces
"""

import sys
import os

def main():
    print("🤖 STRAT Trading Bot Dashboard Launcher")
    print("=" * 50)
    print()
    print("Choose your dashboard:")
    print("1. Basic Dashboard (Original)")
    print("2. Advanced Dashboard (New Features)")
    print("3. Exit")
    print()
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        print("\n🚀 Starting Basic Dashboard...")
        print("Features: Basic pattern detection, simple controls")
        print("Access at: http://localhost:5000")
        print()
        os.system("python app.py")
        
    elif choice == "2":
        print("\n🚀 Starting Advanced Dashboard...")
        print("Features: Watchlist, Sector Analysis, Daily STRAT Results, Multi-Agent System")
        print("Access at: http://localhost:5000")
        print()
        os.system("python advanced_web_interface.py")
        
    elif choice == "3":
        print("👋 Goodbye!")
        sys.exit(0)
        
    else:
        print("❌ Invalid choice. Please try again.")
        main()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Dashboard launcher stopped.")
    except Exception as e:
        print(f"❌ Error: {e}")
        input("Press Enter to exit...")