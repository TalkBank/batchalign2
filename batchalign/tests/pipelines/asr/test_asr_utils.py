from batchalign.pipelines.asr.utils import *
from batchalign.document import *

RAW_OUTPUT = {'monologues': [{'elements': [{'type': 'text', 'ts': 0.42, 'end_ts': 1.12, 'value': 'uh'}, {'type': 'text', 'ts': 1.12, 'end_ts': 1.56, 'value': 'Py'}, {'type': 'text', 'ts': 1.56, 'end_ts': 2.06, 'value': 'audio'}, {'type': 'text', 'ts': 2.06, 'end_ts': 2.46, 'value': 'uh'}, {'type': 'text', 'ts': 2.46, 'end_ts': 2.9, 'value': 'analysis'}, {'type': 'text', 'ts': 2.9, 'end_ts': 3.3, 'value': 'uh'}, {'type': 'text', 'ts': 3.3, 'end_ts': 3.54, 'value': 'Python'}, {'type': 'text', 'ts': 3.54, 'end_ts': 3.92, 'value': 'library'}, {'type': 'text', 'ts': 3.92, 'end_ts': 4.38, 'value': 'covering.'}, {'type': 'text', 'ts': 4.38, 'end_ts': 4.88, 'value': 'uh'}, {'type': 'text', 'ts': 4.88, 'end_ts': None, 'value': 'right'}, {'type': 'text', 'ts': 5.18, 'end_ts': 5.5, 'value': 'range'}, {'type': 'text', 'ts': 5.5, 'end_ts': 5.7, 'value': 'of'}, {'type': 'text', 'ts': 5.7, 'end_ts': 6.08, 'value': 'audio'}, {'type': 'text', 'ts': 6.08, 'end_ts': 6.46, 'value': 'audio'}, {'type': 'text', 'ts': 6.46, 'end_ts': 7.16, 'value': 'analysis'}, {'type': 'text', 'ts': 7.16, 'end_ts': 7.64, 'value': 'tasks'}, {'type': 'text', 'ts': 7.64, 'end_ts': 8.26, 'value': '.'}], 'speaker': 0}]}
ENCODED_OUTPUT = 'uh Py audio uh analysis uh Python library covering . \x15420_4380\x15\nuh right range of audio audio analysis tasks . \x154380_7640\x15'




def test_process_generation():
    processed = process_generation(RAW_OUTPUT, "eng")

    assert str(processed) == ENCODED_OUTPUT
    

