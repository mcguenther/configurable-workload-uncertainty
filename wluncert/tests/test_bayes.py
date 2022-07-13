import sys

sys.path.append('../')
import unittest

from eda4uncert.bayesinference import PyroMCMCRegressor, PyroSVIStructureRegressor, pandas_to_tensor, \
    CompleteContextModelBuilder
from eda4uncert.eda import EDA
from eda4uncert.sws import ConfSysData
from eda4uncert.grammar import InfluenceModelGrammar, PairwiseGrammar, BaseGrammar, Candidate
from fuzzingbook.Fuzzer import RandomFuzzer
from fuzzingbook.Grammars import US_PHONE_GRAMMAR, is_valid_grammar
import logging
import test_basic

test_basic.TestStringMethods().test_train_test_data()

class TestBayesMethods(unittest.TestCase):

    def test_train_test_data(self):
        sys = test_basic.TestStringMethods().test_train_test_data()
        return sys

    def test_structural_svi(self):
        sys = self.test_train_test_data()
        feature_names = sys.opts
        strucReg = PyroSVIStructureRegressor(feature_names=feature_names, n_steps=100, n_samples=100, )
        x_train, x_test, y_train, y_test = sys.test_train_split()
        tensor_x = pandas_to_tensor(x_train)
        tensor_y = pandas_to_tensor(y_train)
        strucReg.fit(tensor_x, tensor_y)
        return strucReg

    def test_full_context_model_builder(self):
        # reg = self.test_structural_svi()
        tps = []
        sys = self.test_train_test_data()
        x_train, x_test, y_train, y_test = sys.test_train_split()
        tensor_x = pandas_to_tensor(x_train)
        tensor_y = pandas_to_tensor(y_train)
        f_names = list(sys.opts)
        model_builder = CompleteContextModelBuilder(f_names)
        model_builder.fit(tensor_x, tensor_y)
        # feature_names = sys.opts
        # tps.extend(reg.get_tuples(feature_names))

    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())
    #
    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)


if __name__ == '__main__':
    unittest.main()
