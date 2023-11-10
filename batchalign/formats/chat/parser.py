from ...document import *
from ...utils import *
from ...errors import *
from ...constants import *
from .lexer import lex, ULTokenType
from ..base import BaseFormat

# FILE = "./extern/minga01a.cha"

class CHATFile(BaseFormat):

    def __init__(self, path=None, document=None):
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

            self.__doc = self.__load(raw)
        else:
            self.__doc = document

    @property
    def doc(self):
        return self.__doc

    @classmethod
    def __load(cls, lines):
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

                # parse the utterance
                tokens = lex(text)

                mor = None
                gra = None
                additional = []

                while raw[0][0] == "%":
                    line = raw.pop(0)
                    beg,line = line.strip()[1:].split(":\t")
                    if beg.strip() == "mor":
                        mor = line
                    elif beg.strip() == "gra":
                        gra = line
                    else:
                        additional.append(CustomLine(id=beg.strip(),
                                                     type=CustomLineType.DEPENDENT,
                                                     content=line.strip()))

                words = list(enumerate(tokens))
                lexed_words = [tok for indx,tok in words if tok[1] == ULTokenType.REGULAR]

                if words[-1][1][0] in ENDING_PUNCT:
                    lexed_words.append(words[-1])
                
                if mor == None:
                    mor = [None for i in range(len(num_lexed_words))]
                else:
                    mor = mor.split(" ")
                if gra == None:
                    gra = [None for i in range(len(num_lexed_words))]
                else:
                    gra = gra.split(" ")

                breakpoint()

                if ((len(mor) != len(gra)) or (len(mor) != len(words)) or
                    (len(words) != len(gra))):
                    raise CHATValidationException(f"Lengths of main, mor, gra tiers are unaligned: lens main={len(words)} mor={len(mor)}, gra={len(gra)}; line: '{text}'")


            # throw error for everything else
            else:
                raise CHATValidationException(f"Unknown line in input CHAT: '{line}'")




