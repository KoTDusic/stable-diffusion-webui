@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS= --xformers --device-id=0 --no-half-vae --skip-prepare

call webui.bat
