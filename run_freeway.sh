#!/bin/sh 
python3 main.py -n_games 1000 -lr 0.00025 -eps_min 0.1 -eps_dec 1e-5 -bs 32 -env 'FreewayDeterministic-v4' -gpu '0' -path 'models/' -algo 'DuelingDDQNAgent'


