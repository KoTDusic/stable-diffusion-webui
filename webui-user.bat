@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--listen --port=9999 --nowebui --force-enable-xformers --device-id=1
set STABLE_DIFFUSION_COMMIT_HASH="c12d960d1ee4f9134c2516862ef991ec52d3f59e"
set ATTN_PRECISION=fp16

call webui.bat
