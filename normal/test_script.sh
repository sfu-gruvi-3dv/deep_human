#!/usr/bin/env bash
python trainer.py -e -m grey_grad_model -t real -sf visualization/grey_grad_model -nb 100
python trainer.py -e -m rgb_model -t real -sf visualization/rgb_model -nb 100
python trainer.py -e -m grey_grad_model -t test -nb 500
python trainer.py -e -m rgb_model -t test -nb 500
