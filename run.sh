#!/bin/bash
python server.py &
sleep 3
python inference.py
wait
