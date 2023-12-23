from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown

from pathlib import Path
import configparser

from batchalign.errors import *

C = Console()

WELCOME = """
# Hello! Welcome to Batchalign!

Thanks so much for checking out **Batchalign**; welcome! This appears to be your first time using Batchalign's command line interface,
so we would like to go through some basic concepts in first-time setup.

For ASR-related tasks, you have two choices of engines to use:

1. Rev.AI, a commercial ASR service in the cloud, or
2. Whisper, a local ASR model

though you can swap between both using flags, Rev requires an API key to function.
"""

REV = """
## Rev.AI Setup
Got it. Let's setup Rev.ai. Please head on over to [Rev signup](https://www.rev.ai/auth/signup) (https://www.rev.ai/auth/signup) to
obtain a API key. Once you are ready, paste it below. We will store it locally on your machine. We will not echo back the
password as you type it, but it will be written down.
"""

FOLDERS = """
## Configuration
FYI, the options you selected during this setup process will be stored in `~/.batchalign.ini` for safekeeping. If you setup
Rev.ai in the previous step, that's where your API key went. Feel free to edit that file for configuration. If you have
questions, please feel free to reach out:

- `macw@cmu.edu`
- `houjun@cmu.edu`
"""


def interactive_setup():
    config = configparser.ConfigParser()
    config["asr"] = {}

    C.print(Markdown(WELCOME))
    configure_rev = Confirm.ask("\nWould you like to set up Rev.ai now?", console=C)

    if configure_rev:
        C.print(Markdown(REV))

        rev_key = Prompt.ask("\nYour Rev.ai API key", console=C)
        rev_key_confirm = Prompt.ask("Just in case, let's do that again. Your Rev.ai key please", console=C)

        while rev_key != rev_key_confirm:
            C.print("\n[italic red]That did not match.[/italic red] Let's try again!\n")
            rev_key = Prompt.ask("Your Rev.ai API key", console=C)
            rev_key_confirm = Prompt.ask("Just in case, let's do that again. Your Rev.ai key please", console=C)

        config["asr"]["engine"] = "rev"
        config["asr"]["engine.rev.key"] = rev_key.strip()
    else:
        config["asr"]["engine"] = "whisper"

    C.print(Markdown(FOLDERS))
    C.print("\n[bold green]Alrighty, let's rock and roll![/bold green] Continuing with Batchalign...\n")

    with open(Path.home()/".batchalign.ini", 'w') as df:
        config.write(df)
    return config

def config_read(interactive=False):
    try:
        with open(Path.home()/".batchalign.ini", 'r+') as df:
            config = configparser.ConfigParser()
            config.read_file(df)

            # dec232023 - patch in Stanza model version as we stopped downloading
            # models unless explicitly requested
            if not config.has_option("ud","model_version"):
                if not config.has_section("ud"):
                    config["ud"] = {}
                config["ud"]["model_version"] = "1.7.0"
                df.seek(0)
                config.write(df)

            return config
    except FileNotFoundError:
        if interactive:
            return interactive_setup()
        else:
            raise ConfigNotFoundError("Batchalign cannot find a configuration file. Run 'batchalign' in the command line to generate one, or write one yourself and place it at `~/.batchalign.ini`.\nHint: did you try to use Rev.AI without supplying an API key?")

