#!/bin/sh 
python3 main.py -n_games 1000 -lr 0.0001 -eps_min 0.01 -eps_dec 1e-5 -bs 32 -env 'BowlingDeterministic-v4' -gpu '0' -path 'models/' -algo 'DDQNAgent'

