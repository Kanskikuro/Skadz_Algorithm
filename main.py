from core.repo import load_champion_priors, load_matchup_data
from ui.app import ChampionPickerGUI

##############################################################################
# 8) Run the GUI
###############################################################################
if __name__ == "__main__":
    # Load precomputed matchup data that includes dedicated columns
    df_matchups = load_matchup_data("data/matchups_shrunk.csv")
    df_priors = load_champion_priors("data/champion_priors.csv")

    # Initialize and run the GUI
    app = ChampionPickerGUI(df_matchups, df_priors)
    app.mainloop()