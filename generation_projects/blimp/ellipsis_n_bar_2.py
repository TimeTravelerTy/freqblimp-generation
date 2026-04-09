from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.randomize import choice, decision
from utils.vocab_sets import *
from utils.vocab_table import get_row_metadata

from generation_projects.blimp.overlay_guards import filter_rows_for_active_zipf

class AgreementGenerator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="syntax",
                         linguistics="ellipsis",
                         uid="ellipsis_n_bar_2",
                         simple_lm_method=True,
                         one_prefix_method=False,
                         two_prefix_method=False,
                         lexically_identical=False)
        self.safe_verbs = all_transitive_verbs
        self.plural_dets = [("two", "three"),
                            ("three", "two"),
                            ("four", "five"),
                            ("five", "four"),
                            ("several", "a few"),
                            ("a few", "several"),
                            ("many", "few"),
                            ("few", "many"),
                            ("a few", "a lot"),
                            ("a lot of", "a few"),
                            ("no", "some"),
                            ("three", "more"),
                            ("three", "fewer"),
                            ("three", "at least as many"),
                            ("three", "almost as many"),
                            ]

        self.singular_dets = [("one", "another"),
                              ("each", "one"),
                              ("this", "another"),
                              ("that", "another"),
                              ("this", "one"),
                              ("that", "one")
                            ]

        # self.singular_dets = [("one", "two"),
        #                         ("one", "five"),
        #                         ("one", "four"),
        #                         ("one", "a few"),
        #                         ("one", "several"),
        #                         ("one", "few"),
        #                         ("one", "many"),
        #                         ("one", "a lot"),
        #                         ("one", "some"),
        #                         ("one", "more"),
        #                         ("one", "at least as many")
        #                     ]

        self.safe_objs_pl = np.setdiff1d(all_nominals, all_proper_names)
        self.safe_objs_sg = np.setdiff1d(all_nominals, all_proper_names)
        self.safe_objs = np.setdiff1d(all_nominals, all_proper_names)

    def _adjective_pool_for_obj(self, obj):
        return filter_rows_for_active_zipf(get_matched_by(obj, "arg_1", all_adjectives), "adjective")

    def _strict_adjective_pool_for_obj(self, obj):
        return filter_rows_for_active_zipf(
            get_matched_by(obj, "arg_1", all_adjectives),
            "adjective",
            fallback_on_empty=False,
        )

    def _source_balanced_pool(self, pool, minimum_size=1):
        base_rows = []
        overlay_rows = []
        for row in pool:
            if get_row_metadata(row).get("source") == "overlay":
                overlay_rows.append(row)
            else:
                base_rows.append(row)
        if len(base_rows) >= minimum_size and len(overlay_rows) >= minimum_size:
            chosen = overlay_rows if decision(0.5) else base_rows
            return np.array(chosen, dtype=pool.dtype)
        return pool

    def sample(self):
        # John  has  had two cups and Jane  has  had three green cups.
        # Subj1 Aux1 V   D1  Obj  AND Subj2 Aux2 V   D2    Adj2  Obj
        # John  has  had two red  cups and Jane  has  had three green.
        # Subj1 Aux1 V   D1  Adj1 Obj  AND Subj2 Aux2 V   D2    Adj2

        V = choice(self.safe_verbs)
        subj_pool = self._source_balanced_pool(get_matches_of(V, "arg_1", all_nominals))
        Subj1 = choice(subj_pool)
        Subj2 = choice(self._source_balanced_pool(get_matches_of(V, "arg_1", all_nominals)), avoid=Subj1)
        Subj1 = N_to_DP_mutate(Subj1)
        Subj2 = N_to_DP_mutate(Subj2)
        Aux1 = return_aux(V, subj=Subj1)
        if is_match_disj(Subj2, Aux1["arg_1"]):
            Aux2 = Aux1
        else:
            Aux2 = return_aux(V, subj=Subj2)
        obj_pool = get_matches_of(V, "arg_2", get_all("mass", "0", self.safe_objs))
        viable_objs = [obj for obj in obj_pool if len(self._strict_adjective_pool_for_obj(obj)) >= 2]
        if viable_objs:
            Obj = choice(self._source_balanced_pool(np.array(viable_objs, dtype=obj_pool.dtype)))
        else:
            Obj = choice(self._source_balanced_pool(obj_pool))
        adjective_pool = self._strict_adjective_pool_for_obj(Obj) if viable_objs else self._adjective_pool_for_obj(Obj)
        adjective_pool = self._source_balanced_pool(adjective_pool, minimum_size=2)
        Adj1 = choice(adjective_pool)
        Adj2 = choice(adjective_pool, avoid=Adj1)
        if Obj["pl"] == "0":
            (D1, D2) = random.choice(self.singular_dets)
        else:
            (D1, D2) = random.choice(self.plural_dets)

        data = {
            "sentence_good": "%s %s %s %s %s and %s %s %s %s %s %s." % (Subj1[0], Aux1[0], V[0], D1, Obj[0], Subj2[0], Aux2[0], V[0], D2, Adj2[0], Obj[0]),
            "sentence_bad": "%s %s %s %s %s %s and %s %s %s %s %s." % (Subj1[0], Aux1[0], V[0], D1, Adj1[0], Obj[0], Subj2[0], Aux2[0], V[0], D2, Adj2[0]),        }
        return data, data["sentence_good"]


def build_generator():
    return AgreementGenerator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
