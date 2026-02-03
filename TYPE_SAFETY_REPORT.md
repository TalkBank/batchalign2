# Type Safety Improvements Report

**Date:** January 31, 2026  
**Scope:** Full codebase type safety audit and remediation  
**Tool:** mypy with `--check-untyped-defs` flag  
**Result:** 80 errors → 0 errors (100% reduction)

---

## Executive Summary

This report documents the comprehensive type safety improvements made to the Batchalign2 codebase. Starting from 80 mypy errors across 20 files, all type issues have been resolved through surgical, targeted fixes. Several actual bugs were discovered and fixed during this process.

---

## Configuration

### mypy.ini
```ini
[mypy]
python_version = 3.11
exclude = batchalign/tests
ignore_missing_imports = True
```

The `--check-untyped-defs` flag was used to enable checking inside functions without type annotations, which revealed significantly more issues than the default configuration.

---

## Errors Fixed by Category

### 1. Wildcard Import Issues (31 errors)

**Problem:** Using `from module import *` prevents mypy from knowing what names are available.

**Files Affected:**
- `batchalign/pipelines/morphosyntax/ud.py` (21 errors)
- `batchalign/pipelines/utterance/ud_utterance.py` (5 errors)
- `batchalign/pipelines/cleanup/parse_support.py` (5 errors)

**Solution:** Replaced wildcard imports with explicit imports:

```python
# Before
from batchalign.document import *
from stanza.models.common.doc import *

# After
from batchalign.document import Document, Utterance, Form, Morphology, Dependency, TokenType, CustomLine
from stanza.models.common.doc import Document as StanzaDocument, Sentence, Token, Word
```

---

### 2. Missing Type Annotations (15 errors)

**Problem:** Variables initialized as empty lists/dicts without type hints caused mypy to infer overly narrow types.

**Files Affected:**
- `batchalign/pipelines/asr/utils.py`
- `batchalign/pipelines/asr/num2chinese.py`
- `batchalign/formats/textgrid/parser.py`
- `batchalign/formats/chat/file.py`
- `batchalign/pipelines/analysis/eval.py`
- `batchalign/models/utterance/prep.py`

**Solution:** Added explicit type annotations:

```python
# Before
final_words = []
tiers = {}
raw = []

# After  
final_words: list[Any] = []
tiers: dict[str, Tier] = {}
raw: list[str] = []
```

---

### 3. Variable Shadowing/Reassignment (12 errors)

**Problem:** Reusing the same variable name with a different type confuses mypy.

**Files Affected:**
- `batchalign/pipelines/morphosyntax/ud.py`
- `batchalign/pipelines/utterance/ud_utterance.py`
- `batchalign/pipelines/morphosyntax/en/irr.py`
- `batchalign/models/utterance/cantonese_infer.py`
- `batchalign/formats/chat/utils.py`

**Solution:** Renamed variables to avoid shadowing:

```python
# Before (irr.py)
proc = [[j.strip() for j in i.split(":")] for i in IRR.strip().split("\n")]
proc = {a.strip(): [k.strip() for k in b.strip().split(",")] for (a,b) in proc}

# After
proc_list = [[j.strip() for j in i.split(":")] for i in IRR.strip().split("\n")]
proc = {a.strip(): [k.strip() for k in b.strip().split(",")] for (a,b) in proc_list}
```

```python
# Before (cantonese_infer.py)
final_passage = []  # list
final_passage = ' '.join(final_passage)  # str

# After
final_passage = []  # list
final_passage_str = ' '.join(final_passage)  # str
```

---

### 4. Method Signature Mismatches (2 errors)

**Problem:** Subclass method signatures didn't match the base class `BatchalignEngine`.

**Files Affected:**
- `batchalign/pipelines/diarization/pyannote.py`
- `batchalign/pipelines/asr/rev.py`

**Solution:** Added proper type hints matching the base class:

```python
# Base class (pipelines/base.py)
def process(self, doc: Document, **kwargs) -> Document:
def generate(self, source_path: str, **kwargs) -> Document:

# Before (pyannote.py)
def process(self, doc):

# After
def process(self, doc: Document, **kwargs) -> Document:
```

---

### 5. None Handling Issues (4 errors)

**Problem:** Calling methods on potentially `None` values without checking.

**Files Affected:**
- `batchalign/pipelines/asr/utils.py`
- `batchalign/formats/textgrid/parser.py`
- `batchalign/models/utils.py`

**Solution:** Added None checks or assertions:

```python
# Before
n2l = NUM2LANG.get(lang.lower())
for a,b in list(reversed(n2l.items())):

# After
n2l = NUM2LANG.get(lang.lower())
if n2l is not None:
    for a,b in list(reversed(n2l.items())):
```

---

### 6. Tuple/Type Confusion Issues (5 errors)

**Problem:** Incorrect type assumptions leading to runtime errors.

**Files Affected:**
- `batchalign/formats/chat/parser.py`
- `batchalign/formats/chat/utils.py`
- `batchalign/pipelines/asr/whisper.py`
- `batchalign/pipelines/asr/oai_whisper.py`

**Solution:** Fixed type handling with proper guards:

```python
# Before (resolve function returns str | tuple[str, str] | None)
res = resolve("whisper", lang)
if res:
    model, base = res  # Error: can't unpack str

# After
res = resolve("whisper", lang)
if res and isinstance(res, tuple):
    model, base = res
```

---

## Actual Bugs Fixed

### Bug 1: Accidental Tuple Creation (CRITICAL)

**File:** `batchalign/models/whisper/infer_asr.py:281`

```python
# Before - creates a tuple due to trailing comma!
current_speaker = element["payload"],

# After - correct assignment
current_speaker = element["payload"]
```

This bug would have caused `current_speaker` to be a tuple `("speaker_1",)` instead of a string `"speaker_1"`, potentially causing speaker recognition failures.

---

### Bug 2: Regex Capturing Group Issue

**File:** `batchalign/formats/chat/parser.py:127`

```python
# Before - capturing group returns tuples
paren_re = re.compile(r"(\[[^\]]*?\])")

# After - non-capturing group returns strings
paren_re = re.compile(r"(?:\[[^\]]*?\])")
```

This bug caused `.strip()` to be called on tuples instead of strings.

---

### Bug 3: Tuple Assigned to List Variable

**File:** `batchalign/formats/chat/utils.py:56`

```python
# Before - zip returns tuples, assigned to list variable
lemmas, feats = zip(*[(i[0], "-".join(i[1:])) for i in feats])

# After - convert to lists
lemmas_tuple, feats_tuple = zip(*[(i[0], "-".join(i[1:])) for i in feats])
lemmas = list(lemmas_tuple)
feats_list = list(feats_tuple)
```

---

## Files Modified

| File | Changes |
|------|---------|
| `batchalign/pipelines/morphosyntax/ud.py` | Replaced wildcard imports, added type annotations |
| `batchalign/pipelines/utterance/ud_utterance.py` | Fixed imports, renamed shadowed variables |
| `batchalign/formats/chat/parser.py` | Fixed regex bug, added type imports |
| `batchalign/models/utils.py` | Added assertions for None checks |
| `batchalign/pipelines/utr/rev_utr.py` | Fixed imports, added client assertion |
| `batchalign/pipelines/cleanup/parse_support.py` | Fixed imports, refactored parsing loop |
| `batchalign/pipelines/asr/num2chinese.py` | Added list type annotation |
| `batchalign/pipelines/asr/utils.py` | Added type annotations, fixed None check |
| `batchalign/formats/textgrid/parser.py` | Added dict type annotations, None guard |
| `batchalign/formats/chat/file.py` | Added list type annotation |
| `batchalign/pipelines/analysis/eval.py` | Added union type annotation |
| `batchalign/models/utterance/prep.py` | Added list type annotations |
| `batchalign/pipelines/morphosyntax/en/irr.py` | Renamed shadowed variable |
| `batchalign/models/utterance/cantonese_infer.py` | Renamed variable to avoid type change |
| `batchalign/models/whisper/infer_asr.py` | **Fixed tuple bug** |
| `batchalign/pipelines/asr/whisperx.py` | Changed None to empty list |
| `batchalign/formats/chat/utils.py` | Fixed tuple/list confusion |
| `batchalign/models/resolve.py` | Added full type annotations |
| `batchalign/pipelines/diarization/pyannote.py` | Fixed method signature |
| `batchalign/pipelines/asr/rev.py` | Fixed method signature, parameter rename |
| `batchalign/pipelines/asr/whisper.py` | Added isinstance guard for tuple unpacking |
| `batchalign/pipelines/asr/oai_whisper.py` | Added isinstance guard for tuple unpacking |

---

## Verification

### mypy Check
```bash
$ python -m mypy batchalign --check-untyped-defs
Success: no issues found in 112 source files
```

### Test Suite
```bash
$ python -m pytest batchalign/tests -x -q
72 passed in 29.44s
```

---

## Recommendations for Future Development

1. **Enable Stricter Checking:** Consider adding these to `mypy.ini`:
   ```ini
   disallow_untyped_defs = True
   warn_return_any = True
   warn_unused_ignores = True
   ```

2. **Avoid Wildcard Imports:** Always use explicit imports for better tooling support and readability.

3. **Don't Reuse Variable Names:** Especially when the type changes; use descriptive suffixes like `_list`, `_str`, `_dict`.

4. **Beware Trailing Commas:** Python treats `x = value,` as a tuple - this is a common source of bugs.

5. **Guard Tuple Unpacking:** When a function can return different types, always check before unpacking.

6. **Add Type Annotations Early:** Especially for function parameters and return types, this catches bugs before they reach production.

---

## Summary Statistics

| Metric | Before | After |
|--------|--------|-------|
| Total Errors | 80 | 0 |
| Files with Errors | 20 | 0 |
| Actual Bugs Found | N/A | 3 |
| Test Suite | 72 passed | 72 passed |

The type safety audit successfully eliminated all mypy errors while discovering and fixing 3 actual bugs that could have caused runtime issues.
