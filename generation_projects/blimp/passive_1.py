from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.randomize import choice

from generation_projects.blimp.overlay_guards import filter_rows_for_active_zipf

class Generator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="syntax",
                         linguistics="argument_structure",
                         uid="passive_1",
                         simple_lm_method=True,
                         one_prefix_method=False,
                         two_prefix_method=False,
                         lexically_identical=False)

        self.en_verbs = get_all("en", "1")
        # Purely intransitive: strict_intrans=1 marks verbs with no transitive frame.
        # This structurally excludes ambitransitive verbs like "contemplate"/"ponder"
        # that carry both frames and whose passives are grammatical.
        self.intransitive = get_all_conjunctive(
            [("strict_intrans", "1"), ("passive", "0")],
            self.en_verbs,
        )
        self.transitive = get_all("passive", "1", self.en_verbs)
        transitive_exprs = set(map(str, self.transitive["expression"]))
        keep_mask = ~np.isin(np.asarray(self.intransitive["expression"], dtype=str), list(transitive_exprs))
        self.intransitive = self.intransitive[keep_mask]
        self.max_sample_attempts = 512

    def sample(self):
        # The girl was attacked by the bear.
        # NP1      be  V_trans  BY NP2
        # The girl was smiled    by the bear.
        # NP1      be  V_intrans BY NP2

        for _ in range(self.max_sample_attempts):
            V_intrans_pool = filter_rows_for_active_zipf(self.intransitive, "verb", fallback_on_empty=False)
            if len(V_intrans_pool) == 0:
                raise LexicalGapError("No xtail-compatible purely-intransitive verb")
            V_intrans = choice(V_intrans_pool)
            NP1_candidates = filter_rows_for_active_zipf(
                get_matches_of(V_intrans, "arg_1", all_nominals), "noun", fallback_on_empty=False
            )
            if len(NP1_candidates) == 0:
                continue
            NP1 = N_to_DP_mutate(choice(NP1_candidates))
            V_trans_pool = filter_rows_for_active_zipf(
                get_matched_by(NP1, "arg_2", self.transitive), "verb", fallback_on_empty=False
            )
            if len(V_trans_pool) == 0:
                continue
            V_trans = choice(V_trans_pool)
            NP2_candidates = filter_rows_for_active_zipf(
                get_matches_of(V_trans, "arg_1", all_nominals), "noun", fallback_on_empty=False
            )
            if len(NP2_candidates) == 0:
                continue
            NP2 = N_to_DP_mutate(choice(NP2_candidates))
            be = return_copula(NP1)
            break
        else:
            raise LexicalGapError("No xtail-compatible passive pair found after bounded retries")

        data = {
            "sentence_good": "%s %s %s by %s." % (NP1[0], be[0], V_trans[0], NP2[0]),
            "sentence_bad": "%s %s %s by %s." % (NP1[0], be[0], V_intrans[0], NP2[0])
        }
        return data, data["sentence_good"]


def build_generator():
    return Generator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
