import os
import unittest

from eda4uncert.bayesinference import PyroMCMCRegressor
from eda4uncert.eda import EDA
from eda4uncert.sws import ConfSysData
from eda4uncert.grammar import InfluenceModelGrammar, PairwiseGrammar, BaseGrammar, Candidate
from fuzzingbook.Fuzzer import RandomFuzzer
from fuzzingbook.Grammars import US_PHONE_GRAMMAR, is_valid_grammar
import logging



class TestStringMethods(unittest.TestCase):
    def test_grammar_validity(self):
        assert is_valid_grammar(US_PHONE_GRAMMAR, supported_opts={'prob'})

    def test_train_test_data(self):
        data_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "training-data",
            "h2.csv",
        )
        sys = ConfSysData(data_path)
        sys.test_train_split()
        return sys

    # def test_model_grammar_validity(self):
    #     g = InfluenceModelGrammar(["A", "B", "C"], max_nonterminals=5)
    #     g_dict = g.get_grammar()
    #     assert is_valid_grammar(g_dict, supported_opts={'prob'})

    def test_create_candidate(self):
        Candidate("A+B+A*C+C*D")

    def test_init(self, grammar: BaseGrammar = InfluenceModelGrammar):
        sys = self.test_train_test_data()
        my_eda = EDA(sys, max_gen=8, grammar=grammar)
        return my_eda

    def test_grammar_graphviz(self):
        my_eda = self.test_init()
        return my_eda.grammar.get_graphviz_grammar_svgs()

    def test_grammar_expansions_df(self):
        my_eda = self.test_init()
        return my_eda.grammar.get_production_df()

    def test_generate_system_individual(self):
        my_eda = self.test_init()
        return my_eda, my_eda.grammar.get_random_ind()

    def test_candidate_get_graphviz(self):
        my_eda, ind = self.test_generate_system_individual()
        ind.get_graphviz(my_eda.grammar)

    def test_scoring(self):
        my_eda, ind = self.test_generate_system_individual()
        my_eda.score(ind)

    def test_init_pop(self):
        my_eda = self.test_init()
        return my_eda, my_eda.generate_pop(200)

    def test_score_pop(self):
        my_eda, pop = self.test_init_pop()
        return my_eda.score_pop(pop)

    def test_truncation(self):
        my_eda, pop = self.test_init_pop()
        top = my_eda.truncation_selection(pop, 0.5)
        return my_eda, top

    def test_grammar_mining(self):
        my_eda, top = self.test_truncation()
        new_grammar = my_eda.grammar.get_updated_grammar(top)
        assert is_valid_grammar(new_grammar.grammar), True

    def test_full_run(self):
        # logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)
        my_eda = self.test_init()
        return my_eda.run(lifes=2, )

    def test_pairwise_inter_grammar(self):
        my_eda = self.test_init(grammar=PairwiseGrammar)
        ind = my_eda.grammar.get_random_ind()
        return my_eda, ind
        # g_dict = g.get_grammar()
        # assert is_valid_grammar(g_dict, supported_opts={'prob'})

    def test_create_p4(self):
        my_eda = self.test_init(grammar=PairwiseGrammar)
        PyroMCMCRegressor(my_eda.grammar)


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
