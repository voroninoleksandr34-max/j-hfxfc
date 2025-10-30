@echo off
echo Stopping Windows Time service...
net stop w32time
echo.
echo Forcing resync with time.windows.com...
w32tm /resync /force
echo.
echo Starting Windows Time service...
net start w32time
echo.
echo Time synchronization complete.
pause