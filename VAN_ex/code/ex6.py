import os

from gtsam.utils import plot

from VAN_ex.code import utils
from VAN_ex.code.Bundle.BundleAdjustment import BundleAdjustment
from VAN_ex.code.DB.DataBase import DataBase

OUTPUT_DIR = 'results/ex6/'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def ex6_run():
    db = DataBase()
    db.read_database(utils.DB_PATH)
    bundle_adjustment = BundleAdjustment(2560, 20, db)
    bundle_adjustment.optimize_all_windows()
    first_window = bundle_adjustment.get_first_window()
    marginals = first_window.get_marginals()
    print(marginals)
    result = first_window.get_current_values()
    plot.plot_trajectory(0, result, marginals=marginals, scale=1, title="Covariance poses for first bundle",
                         save_file=f"{OUTPUT_DIR}Poses rel_covs.png"
                         # , d2_view=False
                         )

if __name__ == '__main__':
    ex6_run()