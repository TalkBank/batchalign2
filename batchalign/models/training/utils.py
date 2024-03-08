"""
training.py
Training operations helpers
"""

import shutil
from dataclasses import dataclass
from typing import Optional, Dict, Any

import rich_click as click
import functools

import os

# common options for batchalign training
def train_func_hydrate(f):
    options = [
        click.argument("run_name",
                       type=str),
        click.argument("data_dir",
                       type=click.Path(file_okay=False)),
        click.argument("model_dir",
                       type=click.Path(file_okay=False)),
        click.option("--wandb",
                     help="Use wandb tracking.",
                     is_flag=True,
                     default=False,
                     type=bool),
        click.option("--wandb_name",
                     help="Wandb name.",
                     default=None,
                     type=str,
                     required=False),
        click.option("--wandb_user",
                     help="Wandb user name.",
                     default=None,
                     required=False,
                     type=str)
    ]

    options.reverse()
    return functools.reduce(lambda x, opt: opt(x), options, f)

def create_config(prep, train, eval, task_name, params, **kwargs):
    wandb = WandbConfig(run_name=(task_name+"_"+kwargs.get("run_name", "")
                                  if not kwargs.get("wandb_name") else kwargs["wandb_name"]),
                        user=kwargs.get("wandb_user", ""))
    proj = Project(task=task_name,
                   prep=prep,
                   train=train,
                   eval=eval)
    config = Config(project=proj,
                    instance_name=kwargs.get("run_name", ""),
                    data_dir=kwargs["data_dir"],
                    model_dir=kwargs["model_dir"],
                    tracker=(None if not kwargs.get("wandb", False) else wandb),
                    params = params)

    return config


@dataclass
class WandbConfig:
    # otherwise defaults to something that the
    # client decides, usually project_name
    run_name: Optional[str]
    # the wandb entity
    user: str

@dataclass
class Project:
    # name of the thing being trained
    task: str
    # one argument which is the Project instance
    prep: callable 
    train: callable
    eval: Optional[callable]

@dataclass
class Config:
    # model information
    project: Project
    # run name
    instance_name: str
    # wandb configuration
    tracker: Optional[WandbConfig]
    # hyperparemeters
    params: Dict[str, any]
    # where the data should be dumped
    data_dir: str
    # where the saved model should be dumped
    model_dir: str

    def resolve_data(self):
        # get data folder
        data_path = os.path.join(self.data_dir, self.project.task, self.instance_name)
        model_path = os.path.join(self.model_dir, self.project.task, self.instance_name)

        os.makedirs(data_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)

        return model_path, data_path, f"{self.project.task}_{self.instance_name}"

def get_clan():
    path = shutil.which("flo") 

    if not path:
        raise RuntimeError("You called a training utility which expects UnixCLAN to parse transcripts.\nHint: visit https://dali.talkbank.org/clan/ to install UnixClan, or run CLAN elsewhere.") 
    else:
        return path
