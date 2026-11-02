@echo off
if not defined FRAMESEEK_PYTHON set "FRAMESEEK_PYTHON=python"
"%FRAMESEEK_PYTHON%" "%~dp0start_backend.py" %*
