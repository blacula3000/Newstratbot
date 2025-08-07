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
    print("2. Advanced Dashboard (Watchlist + Sectors)")
    print("3. STRAT Signals Dashboard (Actionable Signals)")
    print("4. Exit")
    print()
    
    choice = input("Enter your choice (1-4): ").strip()
    
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
        print("\n🎯 Starting STRAT Signals Dashboard...")
        print("Features: Actionable signals with proper STRAT methodology")
        print("• Candlestick classification (1, 2U, 2D, 3)")
        print("• Trigger level break detection")
        print("• Full Time Frame Continuity (FTFC)")
        print("• Real-time signal validation")
        print("Access at: http://localhost:5000/signals")
        print()
        os.system("python advanced_web_interface.py")
        
    elif choice == "4":
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