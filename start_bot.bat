@echo off
echo Starting Trading Bot...
call venv\Scripts\activate
set FLASK_ENV=development
set FLASK_APP=app.web_interface
python -m flask run 