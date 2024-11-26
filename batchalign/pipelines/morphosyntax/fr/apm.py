"""
auditory plural marking for french
"""

from .apmn import APMN

apmn_lits = [
    j.strip()
    for i in APMN.strip().split("\n") if i.strip() != ""
 for j in i.strip().split("\t") if j.strip() != ""
]

def is_apm_noun(n):
    return bool(n.strip().lower() in apmn_lits)

