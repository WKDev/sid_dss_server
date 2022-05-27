#!/bin/sh
# - the script is ran via anaconda python
# - output is not buffered (-u)
# - stdout & stderr are stored to a log file
# - we're executing the working directory specified in the systemd service file
$HOME/anaconda3/bin/python3 -u main.py 2>&1 >> hello.log
sleep 5
google-chrome --kiosk 127.0.0.1:8080
exit 0
