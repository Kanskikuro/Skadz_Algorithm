# Skadz_algorithm
A program that suggests champion picks for league of legends. Skadz, also known as Olav, initiated and layed the foundation for this program, hence naming the project after him as an homage.

## How to run
Pip install the requirement.txt.  
Run the scipts in script. Important notice, run 'dataset.py' before running 'process.py'.
* champion_icon.py
* dataset.py
* process.py

Only then Champ_rec.py can be run.  
To use a shortcut, use the champ_rec.bat.  

## How does the algorithm work
This code implements a champion recommendation and matchup evaluation system for League of Legends.
It works by:
1. Loading synergy and counter data between champions.
2. Loading champion role priors (probabilities of a champion being played in top/jungle/mid/bot/support).
3. Guessing the roles of enemy champions using the Hungarian algorithm (optimal assignment).
4. Calculating expected win probabilities using log-odds from synergy and counter stats.
5. Recommending champions for a given role, while considering: ally team synergy,counters against the enemy and excluded/banned champions

# How does the GUI work
improvements: 
dataset updates, not keep writiting on top of the old dataset.  
dataset and process into one  
champ_rec.bat, use relative location  
