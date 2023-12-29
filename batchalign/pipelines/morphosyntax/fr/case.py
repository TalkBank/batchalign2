"""
special casing rules for french
"""

# For the relative pronoun : "qui" is Nom and "que" is Acc or IO (Indirect Object)
# This is a a bit like the earlier English distinction between "who" and "whom", but it is still alive in French.

# For the interrogative corresponding to English "what" : "qui" is Nom and "que" and "quoi" are both Acc. 

# Personal pronouns involve Case the most

# Nom pronouns are: 1PS je, 2PS tu, 3PS il/elle, 3PP ils/elles
# Acc pronouns are: 1PS me, 2PS te, 3PS le/la  , 3PP les
# IO pronouns are the same as Acc except for 3PS lui and 3PP leur

# There are also disjunctive or strong pronouns that have Nom case. 
# 1PS moi, 2PS toi,  3PS lui/elle, 3PP eux/elles

# Other personal pronouns, such as 1PP nous or 2PP vous don't take Case.

PRON_NOM = ["qui", "je", "tu", "il", "elle", "ils", "elles",
            "moi", "toi", "lui", "elle", "eux", "elles"]
PRON_ACC = ["que", "quoi", "me", "te", "le", "la", "les", "lui", "leur"]

def case(word):
    clean = word.lower().strip("'").strip()
    if clean in PRON_NOM:
        return "Nom"
    elif clean in PRON_ACC:
        return "Acc"
    else:
        return ""

