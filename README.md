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

Varience.py: Currently not completed(20/3/2026)
If use it as an "FPL rules simulator + optimizer": it's essentially complete (auto-subs, formation, captain, and distribution simulation are all included).

If use it as an "end-to-end system (data scraping → training → prediction → transfer + starting XI recommendations)": it still lacks the "integration of models and constraints", so it's not truly ready to run out of the box.
