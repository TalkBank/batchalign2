"""
cli.py
The Batchalign command-line interface
"""

import multiprocessing
import rich_click as click
import functools

import os
from glob import glob

from multiprocessing import Process, freeze_support

from batchalign.pipelines import BatchalignPipeline

from rich.traceback import install
from rich.console import Console
from rich.panel import Panel
from pathlib import Path
from batchalign.document import *
from batchalign.formats.chat import CHATFile
from batchalign.utils import config
from rich.logging import RichHandler

from batchalign.cli.dispatch import _dispatch
from batchalign.models.training.run import cli as train

from enum import Enum

import traceback

import pyfiglet
from rich import pretty
import logging as L 
baL = L.getLogger('batchalign')

C = Console()


with open(Path(__file__).parent.parent / "version", 'r') as df:
    VERSION_NUMBER, RELEASE_DATE, RELEASE_NOTES = df.readlines()[:3]

#################### OPTIONS ################################

# common options for batchalign
def common_options(f):
    options = [
        click.argument("in_dir",
                       type=click.Path(exists=True, file_okay=False)),
        click.argument("out_dir",
                       type=click.Path(exists=True, file_okay=False))
    ]

    options.reverse()
    return functools.reduce(lambda x, opt: opt(x), options, f)

###################### UTILS ##############################

def handle_verbosity(verbosity):
    L.shutdown()
    L.getLogger('stanza').handlers.clear()
    L.getLogger('transformers').handlers.clear()
    L.getLogger('nemo_logger').handlers.clear()
    L.getLogger("stanza").setLevel(L.INFO)
    L.getLogger('nemo_logger').setLevel(L.CRITICAL)
    L.getLogger('batchalign').setLevel(L.WARN)
    L.getLogger('lightning.pytorch.utilities.migration.utils').setLevel(L.ERROR)

    if verbosity >= 2:
        L.basicConfig(format="%(message)s", level=L.ERROR, handlers=[RichHandler(rich_tracebacks=True)])
        L.getLogger('nemo_logger').setLevel(L.INFO)
        L.getLogger('batchalign').setLevel(L.INFO)
    if verbosity >= 3:
        L.getLogger('batchalign').setLevel(L.DEBUG)
    if verbosity >= 4:
        L.getLogger('batchalign').setLevel(L.DEBUG)
        L.getLogger('transformers').setLevel(L.INFO)

@click.group()
@click.pass_context
@click.version_option(VERSION_NUMBER)
@click.option("-v", "--verbose", type=int, count=True, default=0, help="How loquacious Batchalign should be.")
def batchalign(ctx, verbose):
    """process .cha and/or audio files in IN_DIR and dumps them to OUT_DIR using recipe COMMAND"""

    ## setup commands ##
    # multiprocessing thread freeze
    freeze_support()
    # ensure that the context object is a dictionary
    ctx.ensure_object(dict)
    # verbosity
    handle_verbosity(verbose)
    # add to arguments
    ctx.obj["verbose"] = verbose
    # setup config
    ctx.obj["config"] = config.config_read(True)
    # make everything look better
    # pretty.install()
    # better tracebacks
    install()

batchalign.add_command(train, "models")

#################### ALIGN ################################

@batchalign.command()
@common_options
@click.option("--whisper/--rev",
              default=False, help="For utterance timing recovery, OpenAI Whisper (ASR) instead of Rev.AI (default).")
@click.option("--wav2vec/--whisper_fa",
              default=True, help="Use Whisper instead of Wav2Vec for English (defaults for Whisper for non-English)")
@click.option("--iic", is_flag=True, default=False, help="Use IIC forced alignment (for Chinese).")
@click.option("--tencent/--rev",
              default=False, help="Use Tencent instead of Rev.AI (default).")
@click.option("--funaudio/--rev",
              default=False, help="Use FunAudio instead of Rev.AI (default).")
@click.option("--pauses", type=bool, default=False, help="Should we try to bullet each word or should we try to add pauses in between words by grouping them? Default: no pauses.", is_flag=True)
@click.option("--wor/--nowor",
              default=True, help="Should we write word level alignment line? Default to yes.")
@click.pass_context
def align(ctx, in_dir, out_dir, whisper, wav2vec, iic, tencent, funaudio, **kwargs):
    """Align transcripts against corresponding media files."""
    def loader(file):
        return (
            CHATFile(path=os.path.abspath(file)).doc,
            {"pauses": kwargs.get("pauses", False)}
        )

    def writer(doc, output):
        CHATFile(doc=doc).write(output, write_wor=kwargs.get("wor", True))

    # Determine FA engine
    if iic:
        fa_engine = "iic_fa"
    elif not wav2vec:
        fa_engine = "whisper_fa"
    else:
        fa_engine = "wav2vec_fa"

    _dispatch("align", "eng", 1,
              ["cha"], ctx,
              in_dir, out_dir,
              loader, writer, C,
              fa=fa_engine,
              utr = ("whisper_utr" if whisper else
                     ("tencent_utr" if tencent else
                     ("funaudio_utr" if funaudio else "rev_utr"))),
              **kwargs)

#################### TRANSCRIBE ################################

@batchalign.command()
@common_options
@click.option("--whisper_oai/--rev",
              default=False, help="Use the OpenAI's Whisper implementation instead of Rev.AI (default).")
@click.option("--whisper/--rev",
              default=False, help="Use Huggingface's Whisper implementation instead of Rev.AI (default).")
@click.option("--tencent/--rev",
              default=False, help="Use Tencent instead of Rev.AI (default).")
@click.option("--whisperx/--rev",
              default=False, help="Use WhisperX instead of Rev.AI (default). Superceeds --whisper.")
@click.option("--alibaba/--rev",
              default=False, help="Use Alibaba instead of Rev.AI (default). Superceeds --whisper.")
@click.option("--funaudio/--rev",
              default=False, help="Use FunAudio instead of Rev.AI (default). Superceeds --whisper.")
@click.option("--diarize/--nodiarize",
              default=False, help="Perform speaker diarization (this flag is ignored with Rev.AI)")
@click.option("--wor/--nowor",
              default=False, help="Should we write word level alignment line? Default to no.")
@click.option("--lang",
              help="sample language in three-letter ISO 3166-1 alpha-3 code",
              show_default=True,
              default="eng",
              type=str)
@click.option("-n", "--num_speakers", type=int, help="number of speakers in the language sample", default=2)
@click.pass_context
def transcribe(ctx, in_dir, out_dir, lang, num_speakers, **kwargs):
    """Create a transcript from audio files."""
    def loader(file):
        return file

    asr = "rev"
    if kwargs["whisper"]:
        asr = "whisper"
    if kwargs["whisperx"]:
        asr = "whisperx"
    if kwargs["tencent"]:
        asr = "tencent"
    if kwargs["whisper_oai"]:
        asr = "whisper_oai"
    if kwargs["alibaba"]:
        asr = "aliyun"
    if kwargs["funaudio"]:
        asr = "funaudio"

    def writer(doc, output):
        doc.content.insert(0, CustomLine(id="Comment", type=CustomLineType.INDEPENDENT,
                                         content=f"Batchalign {VERSION_NUMBER.strip()}, ASR Engine {asr}. Unchecked output of ASR model."))
        CHATFile(doc=doc).write(output
                                .replace(".wav", ".cha")
                                .replace(".WAV", ".cha")
                                .replace(".mp4", ".cha")
                                .replace(".MP4", ".cha")
                                .replace(".mp3", ".cha")
                                .replace(".MP3", ".cha"),
                                write_wor=kwargs.get("wor", False))

    if kwargs.get("diarize"):
        _dispatch("transcribe_s",
                  lang, num_speakers, ["mp3", "mp4", "wav"], ctx,
                  in_dir, out_dir,
                  loader, writer, C,
                  asr=asr, **kwargs)
    else:
        _dispatch("transcribe",
                  lang, num_speakers, ["mp3", "mp4", "wav"], ctx,
                  in_dir, out_dir,
                  loader, writer, C,
                  asr=asr, **kwargs)

#################### TRANSLATE ################################

@batchalign.command()
@common_options
@click.pass_context
def translate(ctx, in_dir, out_dir, **kwargs):
    """Translate the transcript to English."""

    def loader(file):
        cf = CHATFile(path=os.path.abspath(file), special_mor_=True)
        doc = cf.doc
        # if str(cf).count("%mor") > 0:
        #     doc.ba_special_["special_mor_notation"] = True
        return doc

    def writer(doc, output):
        CHATFile(doc=doc).write(output)

    _dispatch("translate", "eng", 1, ["cha"], ctx,
              in_dir, out_dir,
              loader, writer, C)

#################### MORPHOTAG ################################

@batchalign.command()
@common_options
@click.option("--retokenize/--keeptokens",
              default=False, help="Retokenize the main line to fit the UD tokenizations.")
@click.option("--skipmultilang/--multilang",
              default=False, help="skip code switching")
@click.option("--lexicon",
              type=click.Path(exists=True,
                              file_okay=True, dir_okay=False),
              help="Comma seperated manual lexicon override")
@click.pass_context
def morphotag(ctx, in_dir, out_dir, **kwargs):
    """Perform morphosyntactic analysis on transcripts."""

    def loader(file):
        mwt = {}
        if kwargs.get("lexicon") != None and kwargs.get("lexicon", "").strip() != "":
            import csv
            raw = []
            with open(kwargs["lexicon"], 'r') as df:
                raw = [i for i in csv.reader(df)]
            for i in raw:
                mwt[i[0]] = tuple(i[1:])
        cf = CHATFile(path=os.path.abspath(file), special_mor_=True)
        doc = cf.doc
        if str(cf).count("%mor") > 0:
            doc.ba_special_["special_mor_notation"] = True
        return (
            doc,
            {"retokenize": kwargs["retokenize"],
             "skipmultilang": kwargs["skipmultilang"],
             "mwt": mwt}
        )

    def writer(doc, output):
        CHATFile(doc=doc, special_mor_=doc.ba_special_.get("special_mor_notation", False)).write(output)

    _dispatch("morphotag", "eng", 1, ["cha"], ctx,
              in_dir, out_dir,
              loader, writer, C)


#################### MORPHOTAG ################################

@batchalign.command(hidden=True)
@common_options
@click.pass_context
def coref(ctx, in_dir, out_dir, **kwargs):
    """Perform coreference analysis on transcripts."""

    def loader(file):
        cf = CHATFile(path=os.path.abspath(file))
        doc = cf.doc
        return doc, {}

    def writer(doc, output):
        CHATFile(doc=doc).write(output)

    _dispatch("coref", "eng", 1, ["cha"], ctx,
              in_dir, out_dir,
              loader, writer, C)


#################### UTSEG ################################

@batchalign.command()
@common_options
@click.option("--lang",
              help="sample language in three-letter ISO 3166-1 alpha-3 code",
              show_default=True,
              default="eng",
              type=str)
@click.option("-n", "--num_speakers", type=int, help="number of speakers in the language sample", default=2)
@click.pass_context
def utseg(ctx, in_dir, out_dir, lang, num_speakers, **kwargs):
    """Perform morphosyntactic analysis on transcripts."""

    def loader(file):
        return CHATFile(path=os.path.abspath(file)).doc

    def writer(doc, output):
        CHATFile(doc=doc).write(output)

    _dispatch("utseg", lang, num_speakers, ["cha"], ctx,
              in_dir, out_dir,
              loader, writer, C)

#################### BENCHMARK ################################

@batchalign.command()
@common_options
@click.option("--whisper/--rev",
              default=False, help="Use OpenAI Whisper (ASR) instead of Rev.AI (default).")
@click.option("--tencent/--rev",
              default=False, help="Use Tencent instead of Rev.AI (default).")
@click.option("--funaudio/--rev",
              default=False, help="Use Tencent instead of Rev.AI (default).")
@click.option("--whisper_oai/--rev",
              default=False, help="Use the OpenAI's Whisper implementation instead of Rev.AI (default).")
@click.option("--lang",
              help="sample language in three-letter ISO 3166-1 alpha-3 code",
              show_default=True,
              default="eng",
              type=str)
@click.option("-n", "--num_speakers", type=int, help="number of speakers in the language sample", default=2)
@click.pass_context
def benchmark(ctx, in_dir, out_dir, lang, num_speakers, whisper, tencent, funaudio, whisper_oai, **kwargs):
    """Benchmark ASR utilities for their word accuracy"""
    def loader(file):
        # try to find a .cha in the same directory
        p = Path(file)
        cha = p.with_suffix(".cha")
        # if there are not cha file found, we complain
        if not cha.exists():
            cha = (Path(in_dir)/p.name).with_suffix(".cha")
            if not cha.exists():
                raise FileNotFoundError(f"No gold .cha transcript found, we cannot do benchmarking. audio: {p.name}, desired cha: {cha.name}, looked in: {str(cha)}.")
        # otherwise, load the goald along with the input file
        return file, {"gold": CHATFile(path=str(cha), special_mor_=True).doc}

    def writer(doc, output):
        # delete the copied cha file
        os.remove(Path(output).with_suffix(".cha"))
        # write the wer
        with open(Path(output).with_suffix(".wer.txt"), 'w') as df:
            df.write(str(doc["wer"]))
        with open(Path(output).with_suffix(".diff"), 'w') as df:
            df.write(str(doc["diff"]))
        CHATFile(doc=doc["doc"]).write(str(Path(output).with_suffix(".asr.cha")))


    _dispatch("benchmark", lang, num_speakers, ["mp3", "mp4", "wav"], ctx,
              in_dir, out_dir,
              loader, writer, C,
              asr="whisper" if whisper else ("funaudio" if funaudio else ("tencent" if tencent else ("whisper_oai" if whisper_oai else "rev"))),
              **kwargs)
    

#################### AVQI ################################

@batchalign.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--lang",
              help="sample language in three-letter ISO 3166-1 alpha-3 code",
              show_default=True, default="eng", type=str)
@click.pass_context
def avqi(ctx, input_dir, output_dir, lang, **kwargs):
    """Calculate AVQI from paired .cs and .sv audio files in input directory."""

    from batchalign.pipelines.avqi import AVQIEngine
    from pathlib import Path
    import os
    
    # Get all .cs files
    cs_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if '.cs.' in f and any(f.endswith(ext) for ext in ['.mp3', '.wav', '.mp4']):
                cs_files.append(os.path.join(root, f))
    
    if not cs_files:
        C.print("[bold red]No .cs audio files found in input directory[/bold red]")
        return
    
    C.print(f"\nMode: [blue]avqi[/blue]; got [bold cyan]{len(cs_files)}[/bold cyan] file pair{'s' if len(cs_files) > 1 else ''} to process from {input_dir}:\n")
    
    engine = AVQIEngine()
    
    for cs_file in cs_files:
        cs_path = Path(cs_file)
        C.print(f"Processing: [cyan]{cs_path.name}[/cyan]")
        
        doc = Document.new(media_path=cs_file, lang=lang)
        results = engine.analyze(doc)
        
        # Create output path
        rel_path = os.path.relpath(cs_file, input_dir)
        output_path = Path(os.path.join(output_dir, rel_path))
        os.makedirs(output_path.parent, exist_ok=True)
        
        if results.get('success', False):
            output_txt = output_path.with_suffix('.avqi.txt')
            with open(output_txt, 'w') as f:
                f.write(f"AVQI: {results['avqi']:.3f}\n")
                f.write(f"CPPS: {results['cpps']:.3f}\n")
                f.write(f"HNR: {results['hnr']:.3f}\n")
                f.write(f"Shimmer Local: {results['shimmer_local']:.3f}\n")
                f.write(f"Shimmer Local dB: {results['shimmer_local_db']:.3f}\n")
                f.write(f"LTAS Slope: {results['slope']:.3f}\n")
                f.write(f"LTAS Tilt: {results['tilt']:.3f}\n")
                f.write(f"CS File: {results['cs_file']}\n")
                f.write(f"SV File: {results['sv_file']}\n")
                f.write(f"Language: {lang}\n")
            C.print(f"  [bold green]✓[/bold green] AVQI: {results['avqi']:.3f} → {output_txt.name}")
        else:
            error_file = output_path.with_suffix('.error.txt')
            with open(error_file, 'w') as f:
                f.write(f"AVQI calculation failed: {results.get('error', 'Unknown error')}\n")
            C.print(f"  [bold red]✗[/bold red] Failed: {results.get('error', 'Unknown error')}")
    
    C.print(f"\nAll done. Results saved to {output_dir}!\n")

#################### OPENSMILE ################################

@batchalign.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--feature-set", 
              type=click.Choice(['eGeMAPSv02', 'eGeMAPSv01b', 'GeMAPSv01b', 'ComParE_2016']),
              default='eGeMAPSv02',
              help="Feature set to extract")
@click.option("--lang",
              help="sample language in three-letter ISO 3166-1 alpha-3 code",
              show_default=True, default="eng", type=str)
@click.pass_context
def opensmile(ctx, input_dir, output_dir, feature_set, lang, **kwargs):
    """Extract openSMILE audio features from speech samples."""

    def loader(file):
        doc = Document.new(media_path=file, lang=lang)
        return doc, {"feature_set": feature_set}

    def writer(results, output):
        if results.get('success', False):
            output_csv = Path(output).with_suffix('.opensmile.csv')
            features_df = results.get('features_df')
            if features_df is not None:
                features_df.to_csv(output_csv, header=['value'], index_label='feature')
        else:
            error_file = Path(output).with_suffix('.error.txt')
            with open(error_file, 'w') as f:
                f.write(f"OpenSMILE extraction failed: {results.get('error', 'Unknown error')}\n")

    _dispatch("opensmile", lang, 1, ["mp3", "mp4", "wav"], ctx,
              input_dir, output_dir,
              loader, writer, C, **kwargs)
    
#################### SETUP ################################

@batchalign.command()
@click.pass_context
def setup(ctx):
    """Reconfigure Batchalign settings, such as Rev.AI key."""

    config.interactive_setup()

#################### VERSION ################################

@batchalign.command()
@click.pass_context
def version(ctx, **kwargs):
    """Print program version info and exit."""

    ptr = (pyfiglet.figlet_format("Batchalign2")+"\n" +
           f"Version: [bold]{VERSION_NUMBER.strip()}[/bold], released {RELEASE_DATE.strip()}\n" +
           f"[italic]{RELEASE_NOTES.strip()}[/italic]"+"\n" +
           "\nDeveloped by Brian MacWhinney and Houjun Liu")
    C.print("\n\n"+ptr+"\n\n")
