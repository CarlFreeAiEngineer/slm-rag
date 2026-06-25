@echo off
cd /d "%~dp0"
rem Launches slm-rag with the bundled uv. serve.py downloads the GPU-capable
rem llama.cpp build and the model GGUFs itself on first run if they are missing.
rem Args pass straight through, e.g.  run.bat --cli   or   run.bat --port 8080
"%~dp0bin\uv.exe" run "%~dp0serve.py" %*
