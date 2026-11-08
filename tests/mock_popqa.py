"""Schema-faithful synthetic PopQA table for offline tests (NOT research data)."""
import json
import numpy as np
import pandas as pd

RELATIONS = {  # approx. PopQA relation sizes
    "occupation": 532, "place of birth": 584, "genre": 1619, "father": 570, "country": 838,
    "producer": 1520, "director": 1999, "capital of": 363, "screenwriter": 1999, "composer": 978,
    "color": 34, "religion": 338, "sport": 547, "author": 1514, "mother": 187, "capital": 645,
}
SKEWED = {"sport": ["association football", "basketball", "baseball"], "religion": ["Catholic Church", "Islam"],
          "country": ["United States of America", "France"], "genre": ["pop music", "rock music"],
          "occupation": ["politician", "actor"], "color": ["red", "blue"]}


def make(scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    rows, rid = [], 0
    n_subjects = int(12000 * scale)
    for rel, n in RELATIONS.items():
        for _ in range(max(1, int(n * scale))):
            sid = int(rng.integers(0, n_subjects))
            name = f"Entity {sid}"
            if rel in SKEWED and rng.random() < 0.6:
                obj = SKEWED[rel][int(rng.integers(0, len(SKEWED[rel])))]
            else:
                obj = f"Obj {rel[:3]} {int(rng.integers(0, 400))}"
            oal = [obj.lower(), obj + "s"] if rng.random() < 0.5 else []
            rows.append(dict(id=rid, subj=name, prop=rel, obj=obj, subj_id=sid, prop_id=1, obj_id=hash(obj) % 10**6,
                             s_aliases=json.dumps([name + " Jr"]), o_aliases=json.dumps(oal),
                             s_uri=f"http://www.wikidata.org/entity/Q{sid}", o_uri="", s_wiki_title=name,
                             o_wiki_title=obj, s_pop=int(rng.lognormal(6, 2.5)) + 2, o_pop=10,
                             question=f"What is {name}'s {rel}?", possible_answers=json.dumps([obj] + oal)))
            rid += 1
    df = pd.DataFrame(rows)
    # homonyms: different subj_id, identical question text
    df.loc[df.index[:20], "question"] = "What is John Smith's occupation?"
    df.loc[df.index[:20], "prop"] = "occupation"
    return df
