# ARISE research code. See NOTICE and CITATION.cff.
import hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import torch


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")
    os.replace(str(temp), str(path))


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    os.replace(str(temp), str(path))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        cpu=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all(),
    )


def restore_rng(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["cpu"].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in s["cuda"]])


def progress(out, phase, **kw):
    save_json(
        Path(out) / "progress.json", dict(time=time.time(), pid=os.getpid(), phase=phase, **kw)
    )
