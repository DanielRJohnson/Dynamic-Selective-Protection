from numpy import ndarray
from pandas import read_csv
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.utils._testing import ignore_warnings

from os.path import dirname
from argparse import ArgumentParser
from dataclasses import dataclass, asdict
from joblib import dump
from tqdm import tqdm
from time import strftime


@dataclass
class DataOptions:
    """All options for loading and preparing the data"""
    data_file: str  # ex: ./analyses/data/xyz.csv
    features: list[str]  # ex: ["error_iter", "pos_2norm"]
    target: str  # ex: "slowdown"


@dataclass
class BayesSearchOptions:
    """All options for running the model search"""
    n_jobs: int  # ex: 3
    cv: int  # ex: 5


def get_args() -> tuple[DataOptions, BayesSearchOptions]:
    """Parses a ModelSearchOptions from command line arguments"""
    parser = ArgumentParser(
        description="Runs a model search over a given data file, features, and target.")
    parser.add_argument("--data_file", type=str,
                        help="Path to the data file to train on")
    parser.add_argument("--features", nargs="+", type=str,
                        help="Features in the data file to train on")
    parser.add_argument("--target", type=str,
                        help="Target in the data file to train on")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="see https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html")
    parser.add_argument("--cv", type=int, default=3,
                        help="see https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html")
    args = parser.parse_args()
    return DataOptions(args.data_file, args.features, args.target), \
        BayesSearchOptions(args.n_jobs, args.cv)


def generate_searches(search_opts: BayesSearchOptions) -> list[BayesSearchCV]:
    """
    Generate BayesSearchCV objects for hyperparameter searches for various chosen models.
    """
    # all args but data_file go to BayesSearchCV
    search_kwargs = asdict(search_opts)

    polyreg = make_pipeline(
        RobustScaler(), PolynomialFeatures(), Ridge())
    polyreg_params = {
        "polynomialfeatures__degree": Integer(2, 5),
        "ridge__alpha": Real(1e-8, 0.1, prior="log-uniform")
    }
    polyreg_search = BayesSearchCV(
        polyreg, polyreg_params, **search_kwargs, n_iter=100)

    rf = make_pipeline(RobustScaler(), RandomForestRegressor())
    rf_params = {
        "randomforestregressor__n_estimators": Integer(50, 1000, prior="log-uniform"),
        "randomforestregressor__max_depth": Integer(3, 10)
    }
    rf_search = BayesSearchCV(rf, rf_params, **search_kwargs, n_iter=100)

    xgb = make_pipeline(RobustScaler(), XGBRegressor())
    xgb_params = {
        'xgbregressor__n_estimators': Integer(50, 1000, prior="log-uniform"),
        'xgbregressor__learning_rate': Real(0.001, 0.1),
        'xgbregressor__max_depth': Integer(3, 15),
        'xgbregressor__reg_alpha': Real(0, 0.01),
        'xgbregressor__reg_lambda': Real(0, 0.01),
    }
    xgb_search = BayesSearchCV(xgb, xgb_params, **search_kwargs, n_iter=100)

    knn = make_pipeline(RobustScaler(), KNeighborsRegressor())
    knn_params = {
        "kneighborsregressor__n_neighbors": Integer(2, 10),
        "kneighborsregressor__p": Real(1.0, 2.0)
    }
    knn_search = BayesSearchCV(knn, knn_params, **search_kwargs, n_iter=100)

    svm = make_pipeline(RobustScaler(), PolynomialFeatures(), LinearSVR())
    svm_params = {
        "polynomialfeatures__degree": Integer(2, 5),
        "linearsvr__loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
        "linearsvr__C": Real(1e-6, 1e+6, prior="log-uniform"),
        "linearsvr__epsilon": Real(1e-8, 0.1, prior="log-uniform"),
    }
    svm_search = BayesSearchCV(svm, svm_params, **search_kwargs, n_iter=100)

    return [polyreg_search, rf_search, xgb_search, knn_search, svm_search]


def get_data(opts: DataOptions) -> tuple[ndarray, ndarray]:
    """Extracts data based on the given DataOptions"""
    df = read_csv(opts.data_file)
    X = df[opts.features].to_numpy()
    y = df[opts.target].to_numpy()
    return X, y


# for excessive BayesSearchCV.fit warnings
@ignore_warnings(category=UserWarning)
def run_model_search(searches: list[BayesSearchCV], X: ndarray, y: ndarray) -> None:
    """Runs given searches over data X and y, then dumps best models to file"""
    for search in tqdm(searches, desc="Model Searches"):
        search.fit(X, y)
        best = search.best_estimator_
        dump(best, f"{dirname(__file__)}/analyses/models/best_" +
             f"{best.steps[-1][1].__class__.__name__}_{strftime('%Y_%m_%d-%I_%M_%S_%p')}.pkl")


def main():
    data_opts, search_opts = get_args()
    searches = generate_searches(search_opts)
    X, y = get_data(data_opts)
    run_model_search(searches, X, y)


if __name__ == "__main__":
    main()
