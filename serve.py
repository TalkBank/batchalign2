from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Annotated
import concurrent
import traceback
import asyncio
import uuid
import os

from rich.logging import RichHandler
import logging as L 

L.shutdown()
L.getLogger('stanza').handlers.clear()
L.getLogger('transformers').handlers.clear()
L.getLogger('nemo_logger').handlers.clear()
L.getLogger("stanza").setLevel(L.INFO)
L.getLogger('nemo_logger').setLevel(L.CRITICAL)
L.getLogger('batchalign').setLevel(L.WARN)
L.getLogger('lightning.pytorch.utilities.migration.utils').setLevel(L.ERROR)
L.basicConfig(format="%(message)s", level=L.ERROR, handlers=[RichHandler(rich_tracebacks=True)])
L.getLogger('nemo_logger').setLevel(L.INFO)
L.getLogger('batchalign').setLevel(L.INFO)
L = L.getLogger('batchalign')

from batchalign import BatchalignPipeline, CHATFile

from pathlib import Path
WORKDIR = Path(os.getenv("BA2_WORKDIR", ""))
WORKDIR.mkdir(exist_ok=True)

app = FastAPI(title="TalkBank | Batchalign2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api", response_class=HTMLResponse, include_in_schema=False)
async def home():
    return f"""
    <h1><tt>TalkBank | Batchalign2</tt></h1>

    <pre>
    The JSON API welcomes you. 

    If you see this screen, it is likely that the BA2 API is correctly setup.
    Visit <a href="redoc">here</a> for a programming guide/API specifications.

    If you are expecting to *use* Batchalign, you have ended up in the wrong place.
    Feel free to reach out to houjun@cmu.edu / macw@cmu.edu for help.
    </pre>
    """

def run(id:str, text:list[str], command:str, lang:str):
    workdir = (WORKDIR / id)
    workdir.mkdir(exist_ok=True)

    try:
        pipe = BatchalignPipeline.new(command, lang)
        doc = CHATFile(lines=text).doc
        res = pipe(doc)
        CHATFile(doc=res).write(workdir/"out.cha")

    except Exception:
        exception = traceback.format_exc()
        with open(workdir/"error", 'w') as df:
            df.write(str(exception))

@app.post("/api")
async def submit(
        input: list[UploadFile],
        command: Annotated[str, Form()],
        lang: Annotated[str, Form()],
        background_tasks: BackgroundTasks
):
    """Submit a job for processing."""
    ids = []

    for i in input:
        id = str(uuid.uuid4())
        data = (await i.read()).decode("utf-8").split("\n")
        raw = []
        for value in data:
            if value == "":
                continue
            if value[0] == "\t":
                res = raw.pop()
                res = res.strip("\n") + " " + value[1:]
                raw.append(res)
            else:
                raw.append(value)
        raw = [i.strip() for i in raw]

        background_tasks.add_task(run, id=id, text=raw, command=command, lang=lang)
        ids.append(id)

    return {"payload": ids, "status": "ok", "key": "submitted"}

@app.get("/api/{id}")
async def status(id):
    """Get status of processed job."""

    id = id.strip()
    if not (WORKDIR / id).is_dir():
        return {"key": "not_found", "status": "error", "message": "The requested job is not found."}
    if (WORKDIR / id / "error").is_file():
        with open(str(WORKDIR / id / "error"), 'r') as df:
            return {"key": "job_error", "status": "error", "message": df.read().strip()}
    if not (WORKDIR / id / "out.cha").is_file():
        return {"key": "processing", "status": "pending", "message": "The requested job is still processing."}

    # return FileResponse(WORKDIR / id / "out.cha")
    return {"key": "done", "status": "done", "message": "The requested job is done."}

@app.get("/api/get/{id}.cha")
async def get(id):
    """Get processed job."""

    id = id.strip()
    if not (WORKDIR / id).is_dir():
        return HTTPException(status_code=404, detail="Item not found.")
    if (WORKDIR / id / "error").is_file():
        return HTTPException(status_code=400, detail="Item processing errored.")

    return FileResponse(WORKDIR / id / "out.cha", media_type='application/octet-stream')
