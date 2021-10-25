#! /usr/bin/python3

import mdptoolbox, mdptoolbox.example

P, R = mdptoolbox.example.forest()
fh = mdptoolbox.mdp.FiniteHorizon(P, R, 0.9, 3)
fh.run()
print(fh.V)