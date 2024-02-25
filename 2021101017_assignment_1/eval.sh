#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <argument>"
    exit 1
fi


if [ -z "$1" ]; then
    echo "Argument is empty."
    exit 1
fi


python_script="2.py"

python3 "$python_script" "$1" "12" "2" "3"
