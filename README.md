Team.py: 
Data resources:  https://www.football-data.co.uk/englandm.php
Data included:
Information: Date, HomeTeam, AwayTeam, Referee           
Results: FTHG/FTAG, HTHG/HTAG
           FTR(H/D/A)                   
Calculation: HS/AS, HST/AST, HC/AC     
            HF/AF, HY/AY, HR/AR       
Gambling odds: Bet365, Betway, Pinnacle, William Hill etc.     
            (Win/Draw etc.)  

Data Analysis.py : Doing data analysis based on the data we already have
Data Quality and Availability Check；
Team Strength Modeling；
Fixture Difficulty Proxy；
Rolling Statistics Feasibility Experiment

baseline.py:
RMSE for player-level point prediction (each row is one (player, gameweek) sample)
A simple baseline using the average scores from the last five games
Opponent strength: Utilize the official FPL FDR (team difficulty rating) directly. 
Home/Away: 0/1 code, can interact with opponent strength as a feature.
Run a ridge regression: the model learns about coefficient itself.

Improvement update (14/4/2026)
Notes included
