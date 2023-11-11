from ...document import *
from ...utils import *
from .utils import *
from ...errors import *
from ...constants import *
from .lexer import lex, ULTokenType
from ..base import BaseFormat
import re

# FILE = "./extern/minga01a.cha"

class CHATFile(BaseFormat):

    def __init__(self, path=None, doc=None, lines=None):
        if path:
            # read in the resulting file
            with open(path, "r") as df:
                # get alignment result
                data = df.readlines()
            # conform result with tab-seperated beginnings
            raw = []
            # for each value, if its a tab seperated beginning, we
            # concate it with the previous line
            for value in data:
                if value[0] == "\t":
                    res = raw.pop()
                    res = res.strip("\n") + " " + value[1:]
                    raw.append(res)
                else:
                    raw.append(value)

            self.__doc = self.from_lines(raw).doc
        elif lines:
            self.__doc = self.from_lines(lines).doc
        else:
            self.__doc = doc

    @property
    def doc(self):
        return self.__doc

    @classmethod
    def from_lines(cls, lines):
        raw = lines
        # pop off anything that's before @Begin
        while raw[0].strip() != "@Begin":
            raw.pop(0)
        raw.pop(0)

        results = {
            "content": [], 
            "langs": [], 
            "media": None 
        }

        tiers = {}

        # read data
        while raw[0].strip() != "@End":
            line = raw.pop(0)

            # we throw away participants because there are duplicate
            # info of the same thing in @ID
            if "@Participants" in line or "@Options" in line:
                continue
            # we split because there are multiple languages possible 
            elif "@Languages" in line.strip():
                results["langs"] = [i.strip() for i in line.strip("@Languages:").strip().split(",")]
            # parse participants; the number of | delinates the metedata field
            elif "@ID" in line.strip():
                participant = line.strip("@ID:").strip().split("|")

                tier = Tier(lang=participant[0], corpus=participant[1], 
                            id=participant[2], name=participant[7])
                tiers[participant[2]] = tier
            # parse media type
            elif "@Media" in line.strip():
                type = MediaType.UNLINKED_AUDIO

                if "unlinked" in line and "audio" in line:
                    type = MediaType.UNLINKED_AUDIO
                elif "unlinked" in line and "video" in line:
                    type = MediaType.UNLINKED_VIDEO
                elif "unlinked" in line and "audio" in line:
                    type = MediaType.AUDIO
                elif "unlinked" in line and "video" in line:
                    type = MediaType.VIDEO

                media = line.strip("@Media:").split(",")
                results["media"] = Media(type=type, name=media[0].strip(),
                                         url=None)
            # depenent tiers with @ are counted as "other" and are inserted as-is
            elif line.strip()[0] == "@":
                beg,end = line.strip()[1:].split(":")
                line = CustomLine(id=beg.strip(),
                                  type=CustomLineType.INDEPENDENT,
                                  content=end.strip())
                results["content"].append(line)
            # we now parse main tiers
            elif line.strip()[0] == "*":
                utterance = {}
                tier,text = line.strip()[1:].split(":\t")
                utterance["text"] = text

                # parse mor and gra lines, and append all other tiers
                mor = None
                gra = None
                wor = None
                additional = []

                while raw[0][0] == "%":
                    line = raw.pop(0)
                    beg,line = line.strip()[1:].split(":\t")
                    if beg.strip() == "mor":
                        mor = line
                    elif beg.strip() == "gra":
                        gra = line
                    elif beg.strip() == "wor" or beg.strip() == "xwor":
                        wor = line
                    else:
                        additional.append(CustomLine(id=beg.strip(),
                                                     type=CustomLineType.DEPENDENT,
                                                     content=line.strip()))

                # parse the actual utterance
                parsed, delim = cls.parse_utterance(text, mor, gra, wor, additional)

                # get the timing of the utterance
                try:
                    t = tiers[tier]
                except KeyError:
                    raise CHATValidationException(f"Encountered undeclared tier: tier='{tier}', line='{text}'")
                ut = Utterance.model_validate({
                    "tier": t,
                    "content": parsed,
                    "text": text,
                    "delim": delim,
                    "custom_dependencies": additional
                })

                timing = re.findall(rf"\x15(\d+)_(\d+)\x15", text)
                if len(timing) != 0:
                    x,y = timing[0]
                    ut.alignment = (int(x), int(y))
                results["content"].append(ut)

            # throw error for everything else
            else:
                raise CHATValidationException(f"Unknown line in input CHAT: '{line}'")

        doc = Document.model_validate(results)
        return cls(doc=doc)

    @staticmethod
    def parse_utterance(text, mor, gra, wor, additional):
        """Encode an utterance into a Batchalign utterance.

        Parameters
        ----------
        text : str
            String text of the main tier.
        mor : Optional[str]
            mor line, or None.
        gra : Optional[str]
            gra line, or None.
        wor : Optional[str]
            wor line, or None.
        additional : List[CustomLine]
            Any custom lines.

        Returns
        -------
        Utterance, str
            Parsed utterance and the utterance delimiter.

        Raises
        ------
        CHATValidationException
            If the utterance didn't parse corrected, we raise an
            exception describing the issue.
        """

        # lex the utterance
        tokens = lex(text)

        # seperate out main words by whether it should have phonation/morphology and add ending punct
        words = list(enumerate(tokens))
        lexed_words = [tok for tok in words if tok[1][1] in [ULTokenType.REGULAR,
                                                                ULTokenType.MORPUNCT]]
        phonated_words = [tok for tok in words if tok[1][1] in [ULTokenType.REGULAR,
                                                                ULTokenType.RETRACE,
                                                                ULTokenType.MORPUNCT,
                                                                ULTokenType.FP]]
        # create base forms
        parsed_forms = {indx:Form(text=text) for indx, (text, _) in phonated_words}

        # stamp an eding punct if its not there
        delim = "."
        if words[-1][1][0] in ENDING_PUNCT:
            lexed_words.append(words[-1])
            parsed_forms[words[-1][0]] = Form(text=words[-1][1][0])
            delim = words[-1][1][0]

        # parse mor gra wor lines
        if mor == None:
            mor = [None for i in range(len(lexed_words))]
        else:
            mor = mor.split(" ")
        if gra != None:
            gra = gra.split(" ")
        if wor == None:
            wor = [None for i in range(len(phonated_words))]
        else:
            words = re.findall(rf"[^{''.join(ENDING_PUNCT)} ]+ ?(\x15\d+_\d+\x15)?", wor)
            wor = []
            for i in words:
                if i.strip() == "":
                    wor.append(None)
                    continue
                x, y = re.findall(r"\d+", i)
                wor.append([int(x),int(y)])

        # check lengths
        if len(lexed_words) != len(mor):
            raise CHATValidationException(f"Lengths of main and mor tiers are unaligned: lens main (filtered for morphology)={len(lexed_words)} mor={len(mor)}; line: '{text}'")
        if len(phonated_words) != len(wor):
            raise CHATValidationException(f"Lengths of main and wor tiers are unaligned: lens main (filtered for phonation)={len(phonated_words)} mor={len(wor)}; line: '{text}'")

        # insert morphology into the parsed forms
        for (indx, _), m in zip(lexed_words, mor):
            if m:
                mor = parse_mor(m)
                parsed_forms[indx].morphology = mor

                # grab the right lines in dependency
                if gra != None:
                    deps = []
                    for _ in range(len(mor)):
                        try:
                            deps.append(gra.pop(0))
                        except IndexError:
                            raise CHATValidationException(f"Lengths of mor and gra tiers are misaligned --- gra line too short on line: '{text}'")
                    if all(i != None for i in deps):
                        deps = [parse_gra(i) for i in deps]
                        parsed_forms[indx].dependency = deps

        if all([i != None for i in mor]) and gra != None and len(gra) != 0:
            raise CHATValidationException(f"Lengths of mor and gra tiers are misaligned --- gra line too long on line: '{text}'")

        # insert phonation into the parsed forms
        for (indx, _), w in zip(phonated_words, wor):
            if w:
                parsed_forms[indx].time = tuple(w)

        # return final parsed forms
        forms = [i[1] for i in sorted(parsed_forms.items(), key=lambda x:x[0])]

        return forms, delim


