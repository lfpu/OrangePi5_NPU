#!/bin/bash
# This script is used to start the RKNN server and ensure it runs correctly.
RUN chmod +x /usr/bin/rknn_server
RUN chmod +x /usr/bin/start_rknn.sh
RUN chmod +x /usr/bin/restart_rknn.sh
RUN restart_rknn.sh

exec uvicorn main:app --host 0.0.0.0 --port 8000