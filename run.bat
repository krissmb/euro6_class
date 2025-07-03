@echo off
REM Путь к виртуальному окружению — если venv в текущей папке, так:
call venv\Scripts\activate.bat

REM Запускаем detect.py
python detect.py

REM Ждём нажатия клавиши, чтобы окно не закрылось
pause
