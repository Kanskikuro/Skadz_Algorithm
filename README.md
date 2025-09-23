# Skadz_algorithm
A program that suggests champion picks for league of legends. Skadz, also known as Olav, initiated and layed the foundation for this program, hence naming the project after him as an homage.

## How to run
Pip install the requirement.txt.  
Run the scipts.py in script

## How does the algorithm work
This code implements a champion recommendation and matchup evaluation system for League of Legends.
It works by:
1. Loading synergy and counter data between champions.
2. Loading champion role priors (probabilities of a champion being played in top/jungle/mid/bot/support).
3. Guessing the roles of enemy champions using the Hungarian algorithm (optimal assignment).
4. Calculating expected win probabilities using log-odds from synergy and counter stats.
5. Recommending champions for a given role, while considering: ally team synergy,counters against the enemy and excluded/banned champions
