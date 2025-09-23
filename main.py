from core.repo import MatchupRepository, PriorsRepository

from ui.app import ChampionPickerGUI

##############################################################################
# 8) Run the GUI
###############################################################################
if __name__ == "__main__":
    # Load precomputed matchup data that includes dedicated columns
    dfm = MatchupRepository.from_csv("data/matchups_shrunk.csv")
    fdp = PriorsRepository.from_csv("data/champion_priors.csv")

    # Initialize and run the GUI
    app = ChampionPickerGUI(dfm, fdp)
    app.mainloop()