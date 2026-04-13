from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.randomize import choice

from generation_projects.blimp.overlay_guards import filter_rows_for_active_zipf

class Generator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="semantics",
                         linguistics="npi_licensing",
                         uid="only_npi_licensor_present",
                         simple_lm_method=True,
                         one_prefix_method=False,
                         two_prefix_method=True,
                         lexically_identical=False)
        self.safe_verbs = np.setdiff1d(all_verbs, all_ing_verbs)

    def sample(self):
        # Only the drivers should ever love the dancers
        # ONLY subj        aux    EVER VP
        # Even the drivers should ever love the dancers
        # EVEN subj        aux    EVER VP

        V = choice(filter_rows_for_active_zipf(self.safe_verbs, "verb"))
        bad_quantifiers = ["all", "every", "each", "most", "many", "a lot of"]
        args = verb_args_from_verb(V, allow_negated=False)
        n_tries = 0
        while any(args["subj"]["expression"].startswith(x) for x in bad_quantifiers) and n_tries < 20:
            args = verb_args_from_verb(V, allow_negated=False)
            n_tries += 1
        if n_tries == 20:
            raise ValueError
        VP = V_to_VP_mutate(V, aux=False, args=args)

        data = {
            "sentence_good": "Only %s %s ever %s." % (args["subj"][0], args["aux"][0], VP[0]),
            "sentence_bad": "Even %s %s ever %s." % (args["subj"][0], args["aux"][0], VP[0]),
            "two_prefix_prefix_good": "Only %s %s" % (args["subj"][0], args["aux"][0]),
            "two_prefix_prefix_bad": "Even %s %s" % (args["subj"][0], args["aux"][0]),
            "two_prefix_word": "ever"
        }
        return data, data["sentence_good"]


def build_generator():
    return Generator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
