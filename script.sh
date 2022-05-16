#!/bin/bash
for line in $(cat requirements3.txt)
do
  pip install $line
done
