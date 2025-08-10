#!/usr/bin/env python3
"""
Simple Local Setup Script - Windows Compatible
Prepares the environment for running the enhanced agent system locally.
"""

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"[*] {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"[+] {description} completed successfully")
        if result.stdout.strip():
            print(f"    Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[-] {description} failed:")
        if e.stderr:
            print(f"    Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("[*] Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"[+] Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"[-] Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("    Minimum required: Python 3.8+")
        return False

def install_dependencies():
    """Install required packages"""
    print("[*] Installing dependencies...")
    
    # Install core packages first
    core_packages = ["numpy", "pandas", "flask", "yfinance"]
    
    for package in core_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"[-] Failed to install {package}, continuing...")
    
    # Install from requirements file
    if os.path.exists("requirements.txt"):
        return run_command("pip install -r requirements.txt", "Installing all requirements")
    else:
        print("[-] requirements.txt not found, installing basic packages")
        basic_packages = [
            "flask==2.3.3",
            "pandas==2.1.1", 
            "numpy==1.24.3",
            "yfinance==0.2.22",
            "ccxt==3.0.0"
        ]
        for package in basic_packages:
            run_command(f"pip install {package}", f"Installing {package}")
        return True

def verify_installation():
    """Verify that key components can be imported"""
    print("[*] Verifying installation...")
    
    test_imports = [
        'pandas',
        'numpy', 
        'flask',
        'yfinance'
    ]
    
    failed_imports = []
    
    for module in test_imports:
        try:
            __import__(module)
            print(f"    [+] {module}")
        except ImportError as e:
            print(f"    [-] {module}: {e}")
            failed_imports.append(module)
    
    if not failed_imports:
        print("[+] All core dependencies verified successfully!")
        return True
    else:
        print(f"[-] Failed imports: {', '.join(failed_imports)}")
        return False

def create_startup_script():
    """Create a simple startup script"""
    script_content = '''@echo off
echo Starting Newstratbot Enhanced Agents...
echo.
python run_local_agents.py
pause
'''
    
    try:
        with open('start_local.bat', 'w') as f:
            f.write(script_content)
        print("[+] Created start_local.bat")
        return True
    except Exception as e:
        print(f"[-] Failed to create startup script: {e}")
        return False

def main():
    """Main setup process"""
    print("Newstratbot Enhanced Agents - Local Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("[-] Setup failed: Incompatible Python version")
        input("Press Enter to exit...")
        return False
    
    # Install dependencies
    print("\n[*] Installing dependencies...")
    install_dependencies()
    
    # Verify installation
    print("\n[*] Verifying installation...")
    verify_installation()
    
    # Create startup script
    print("\n[*] Creating startup scripts...")
    create_startup_script()
    
    print("\n" + "="*50)
    print("SETUP COMPLETE!")
    print("="*50)
    print("\nNEXT STEPS:")
    print("1. Run: python run_local_agents.py")
    print("   Or double-click: start_local.bat")
    print("2. Select Demo Mode to see agent capabilities")
    print("3. Review the logs in local_agents.log")
    print("\nFor documentation, see:")
    print("- AGENT_ARCHITECTURE_GUIDE.md")
    print("- AGENT_INTEGRATION_QUICKSTART.md")
    
    print(f"\nSetup completed successfully!")
    input("Press Enter to exit...")
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
    except Exception as e:
        print(f"\nSetup failed with error: {e}")
        input("Press Enter to exit...")