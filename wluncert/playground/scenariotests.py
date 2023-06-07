import unittest
from uncerteval import Jump3rDataProvider, StandardizerByTrainSet, CompletePoolingModel, ModelFitter, NoPoolingModel, \
    MultilevelLassoModel, DataProvider
import pandas as pd
from sklearn.metrics import r2_score
import arviz as az
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_multilevel_accuracy(self):
        jump3rdf = pd.read_csv("training-data/jump3r.csv")
        jprov = Jump3rDataProvider(jump3rdf, train_number_abs=100)
        standardizer = StandardizerByTrainSet(jprov)
        cpool_model = CompletePoolingModel()
        cpool_fitter = fit_model(cpool_model, standardizer)
        nopool_model = NoPoolingModel()
        nopool_fitter = fit_model(nopool_model, standardizer)
        multilevel_model = MultilevelLassoModel()
        multilevel_fitter = fit_model(multilevel_model, standardizer)
        print("done")


if __name__ == '__main__':
    unittest.main()


def eval_r2(fitter:ModelFitter, data_provider:DataProvider):
    test = data_provider.get_test_data()
    test_y = list(test.iloc[:,-1])
    y_pred_scalar = fitter.predict(test)
    r2 = r2_score(test_y, y_pred_scalar)
    return r2


def fit_model(model, data_provider, rnd=0):
    fitter = ModelFitter(model)
    train_data = data_provider.get_train_data()
    fitter.fit(train_data, rnd=rnd)
    return fitter


def oracle(lin, const, switch, irr, wl_length, wl_mono):
    base = 1
    infl_lin = 0.2 * wl_length
    infl_const = 2
    infl_switch = 0.1 * wl_length if not wl_mono else 0.0
    infl_irr = 0.01
    perf = base + infl_lin * lin + infl_const * const + infl_switch * switch + infl_irr * irr
    return perf
