# ARISE research code. See NOTICE and CITATION.cff.
"""Standalone finite worker: D0 select; D1 test/refit; D2 test/refit; D3 test."""

import json, time
import numpy as np
import pandas as pd
import torch
from .config import ROOT, Config, SEEDS, DATASETS, METHODS, ABLATIONS
from .data import Environment, pack, subset, public_view
from .models import ARISE, CoupledWindowMixer
from .training import Bundle, fit, evaluate, cpu_state
from .memory import Memory, read
from .cache import FrozenCache
from .metrics import NAMES, macro
from .util import save_json, atomic_torch, seed_all, progress, rng_state, restore_rng, sha


class Worker:
    def __init__(self, dataset, method, seed):
        if dataset not in DATASETS or method not in METHODS + ABLATIONS or seed not in SEEDS:
            raise ValueError("Use a registered core dataset, method and training seed")
        if method != "full":
            reference = ROOT / "results" / dataset / str(seed) / "full"
            if not all(
                (reference / name).is_file() for name in ["selection.json", "done.json"]
            ):
                raise ValueError(
                    "Complete the same-seed ARISE reference before a component removal"
                )
        if method.startswith("no_") and dataset != "ml1m":
            raise ValueError("Component removals are supported only for the ML-1M basic study")
        self.name = dataset
        self.method = method
        self.seed = seed
        self.cfg = Config(seed=seed)
        self.out = ROOT / "results" / dataset / str(seed) / method
        self.out.mkdir(parents=True, exist_ok=True)
        import fcntl

        self.lock = (self.out / "worker.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.env = Environment(dataset, "cuda:0")
        self.started = time.time()
        self.no_cold = method == "no_cold"
        self.no_adaptive = method in ["no_adaptive", "no_adaptive_cooc"]
        self.no_cooc = method in ["no_cooc", "no_adaptive_cooc"]
        self.li = 0.0
        self.scale = 0.0
        self.window = 10
        self.records = []
        seed_all(seed)
        self.bundle = Bundle(ARISE(self.env, self.cfg), CoupledWindowMixer(self.cfg)).cuda()
        self.mem = Memory(self.env.nitems, with_ct=not self.no_cooc)
        save_json(
            self.out / "protocol.json",
            dict(
                dataset=dataset,
                method=method,
                seed=seed,
                config=self.cfg.dump(),
                tag_order=self.env.spec["tag_order"],
                tag_types=self.env.spec["tag_sizes"],
                feedback_calibration=False,
                interest_and_address=dataset == "yelp",
                candidate_protocol=self.env.spec,
                release=json.loads((ROOT / "configs/release.json").read_text())["version"],
            ),
        )

    def caches(self, d, window):
        return {
            c: FrozenCache(self.bundle.model, self.env, d, c, window, self.no_cold, self.out)
            for c in ["clean", "deleted60", "replaced60"]
        }

    def enable(self, d, mask):
        ids = np.unique(np.r_[d["targets"][mask].ravel(), d["negatives"][mask].ravel()])
        old = self.bundle.model.rho.detach().cpu().numpy().copy()
        self.bundle.model.enable(ids, d=d, mask=mask)
        return ids[~old[ids]]

    def snapshot(self, path):
        atomic_torch(
            path,
            dict(
                model=cpu_state(self.bundle),
                rng=rng_state(),
                li=self.li,
                scale=self.scale,
                window=self.window,
            ),
        )

    def restore(self, path):
        s = torch.load(path, map_location="cpu")
        self.bundle.load_state_dict(s["model"])
        restore_rng(s["rng"])
        self.li = s["li"]
        self.scale = s["scale"]
        self.window = s["window"]

    @torch.no_grad()
    def calibrate(self, d, R):
        model = self.bundle.model
        model.eval()
        e = model.items()
        users = np.unique(d["round_user"][d["training"]])
        ru = np.bincount(d["round_user"][d["training"]], minlength=len(d["user_ids"]))
        sums = np.zeros(2)
        lookup = np.full(self.env.nitems, -1, np.int64)
        lookup[d["catalog"]] = np.arange(len(d["catalog"]))
        for off in range(0, len(users), self.cfg.user_batch):
            p = pack(d, users[off : off + self.cfg.user_batch], d["training"])
            for start in range(0, len(p["groups"]), self.cfg.group_chunk):
                q = subset(
                    p, np.arange(start, min(start + self.cfg.group_chunk, len(p["groups"])))
                )
                g = q["groups"]
                pos = d["targets"][g]
                neg = d["negatives"][g]
                (v, _, _) = model.user(e, public_view(d), q, "clean", no_cold=self.no_cold)
                pp = torch.as_tensor(pos.astype(np.int64), device=e.device)
                nn = torch.as_tensor(neg.astype(np.int64), device=e.device)
                sp = (v[:, None] * e[pp]).sum(-1) / self.cfg.tau
                sn = (v[:, None] * e[nn]).sum(-1) / self.cfg.tau
                c = read(q, R, 50)
                m0 = (sp[:, :, None] - sn[:, None, :]).square().mean((1, 2)).cpu().numpy()
                m1 = (
                    np.take_along_axis(c, lookup[pos], 1)[:, :, None]
                    - np.take_along_axis(c, lookup[neg], 1)[:, None, :]
                ) ** 2
                sums[0] += (m0 / ru[d["round_user"][g]]).sum()
                sums[1] += (m1.mean((1, 2)) / ru[d["round_user"][g]]).sum()
        sig = np.sqrt(sums / len(users))
        self.scale = float(sig[0] / sig[1]) if sig[1] > 1e-12 else 0.0
        save_json(
            self.out / "scale.json",
            dict(
                native_RMS=float(sig[0]),
                CT_RMS=float(sig[1]),
                scale=self.scale,
                D0_training_only=True,
                user_normalized=True,
            ),
        )

    def d0(self):
        checkpoint = self.out / "D0.pt"
        if checkpoint.exists():
            self.restore(checkpoint)
            self.mem = Memory.load(self.out / "memory_D0")
            return
        d = self.env.stage(0)
        mask = d["training"]
        self.enable(d, mask)
        fit(
            self.bundle,
            self.env,
            d,
            mask,
            self.out,
            "D0_theta",
            "theta",
            self.cfg.d0_epochs,
            no_cold=self.no_cold,
        )
        if not self.no_cold:
            fit(
                self.bundle,
                self.env,
                d,
                mask,
                self.out,
                "D0_adapter",
                "adapter",
                self.cfg.d0_epochs,
            )
        self.mem.update(d, mask)
        R = self.mem.kernel(d["catalog"])
        self.calibrate(d, R)
        if self.method != "full":
            ref = ROOT / "results" / self.name / str(self.seed) / "full/selection.json"
            a = json.loads(ref.read_text())
            self.scale = a["scale"]
            self.li = 0.0 if self.no_cooc else a["lambda_I"]
            self.window = a["short_window"]
        else:
            grid = []
            for li in [0.0, 0.1, 0.25, 0.5]:
                v = evaluate(
                    self.bundle,
                    self.env,
                    d,
                    d["validation"],
                    "clean",
                    self.out,
                    scope="theta",
                    R=R,
                    li=li,
                    scale=self.scale,
                )
                a = macro(v, d, d["validation"])
                j = np.mean([a[p]["ndcg10"] for p in ["first", "follow"]])
                grid.append(dict(lambda_I=li, objective=float(j)))
            self.li = max(grid, key=lambda z: z["objective"])["lambda_I"]
            save_json(self.out / "lambda_grid.json", grid)
        if not self.no_adaptive:
            windows = [10, 20, 50] if self.method == "full" else [self.window]
            base = cpu_state(self.bundle)
            scores = []
            for w in windows:
                self.bundle.load_state_dict(base)
                caches = self.caches(d, w)
                anchor = cpu_state(self.bundle)
                a = fit(
                    self.bundle,
                    self.env,
                    d,
                    mask,
                    self.out,
                    "D0_cwm_W%d" % w,
                    "cwm",
                    self.cfg.d0_epochs,
                    R=R,
                    li=self.li,
                    scale=self.scale,
                    window=w,
                    cache=caches,
                    anchor=anchor,
                    stable=self.cfg.gate_stability,
                )
                scores.append(dict(window=w, objective=a["selected_objective"]))
            self.window = max(scores, key=lambda z: z["objective"])["window"]
            path = self.out / "training" / ("D0_cwm_W%d" % self.window) / "best.pt"
            s = torch.load(path, map_location="cpu")
            self.bundle.load_state_dict(s["model"])
            restore_rng(s["rng"])
            save_json(self.out / "window_grid.json", scores)
        save_json(
            self.out / "selection.json",
            dict(
                lambda_I=self.li,
                scale=self.scale,
                short_window=self.window,
                D0_only=True,
                feedback_calibration=False,
            ),
        )
        self.mem.save(self.out / "memory_D0")
        self.snapshot(checkpoint)

    def publish(self, d, v):
        stage = d["stage"]
        frame = pd.DataFrame(v, columns=NAMES)
        frame.insert(0, "observed_count", 5 * d["round_index"])
        frame.insert(0, "round", d["round_index"] + 1)
        frame.insert(0, "user_id", d["user_ids"][d["round_user"]])
        frame.insert(0, "condition", "clean")
        frame.insert(0, "stage", "D%d" % stage)
        frame.insert(0, "seed", self.seed)
        frame.insert(0, "method", self.method)
        frame.insert(0, "dataset", self.name)
        frame.to_csv(
            self.out / ("D%d_users_rounds.csv.gz" % stage), index=False, compression="gzip"
        )
        summary = macro(v, d)
        save_json(self.out / ("D%d_metrics.json" % stage), summary)
        for part, a in summary.items():
            self.records.append(
                dict(
                    dataset=self.name,
                    method=self.method,
                    seed=self.seed,
                    stage="D%d" % stage,
                    condition="clean",
                    part=part,
                    **a,
                )
            )
        pd.DataFrame(self.records).to_csv(self.out / "metrics.csv", index=False)

    def update(self, d):
        stage = d["stage"]
        checkpoint = self.out / ("D%d.pt" % stage)
        if checkpoint.exists():
            self.restore(checkpoint)
            self.mem = Memory.load(self.out / ("memory_D%d" % stage))
            return
        entry = self.out / ("D%d_entry.pt" % stage)
        if not entry.exists():
            self.snapshot(entry)
        self.restore(entry)
        original = cpu_state(self.bundle)
        new = self.enable(d, d["training"])
        R = self.mem.kernel(d["catalog"])
        scopes = (
            ["theta"]
            + ([] if self.no_cold else ["adapter"])
            + ([] if self.no_adaptive else ["cwm"])
        )
        budgets = {}
        for scope in scopes:
            kwargs = dict(no_cold=self.no_cold)
            if scope == "cwm":
                kwargs.update(
                    R=R,
                    li=self.li,
                    scale=self.scale,
                    window=self.window,
                    cache=self.caches(d, self.window),
                )
            a = fit(
                self.bundle,
                self.env,
                d,
                d["training"],
                self.out,
                "select_D%d_%s" % (stage, scope),
                scope,
                self.cfg.incremental_epochs,
                anchor=original,
                stable=0.0 if scope == "adapter" else self.cfg.stability,
                new_ids=new,
                **kwargs,
            )
            budgets[scope] = a["selected_epoch"]
        self.restore(entry)
        original = cpu_state(self.bundle)
        allmask = np.ones(len(d["targets"]), bool)
        new = self.enable(d, allmask)
        for scope in scopes:
            kwargs = dict(no_cold=self.no_cold)
            if scope == "cwm":
                kwargs.update(
                    R=R,
                    li=self.li,
                    scale=self.scale,
                    window=self.window,
                    cache=self.caches(d, self.window),
                )
            fit(
                self.bundle,
                self.env,
                d,
                allmask,
                self.out,
                "refit_D%d_%s" % (stage, scope),
                scope,
                budgets[scope],
                validation=False,
                anchor=original,
                stable=0.0 if scope == "adapter" else self.cfg.stability,
                new_ids=new,
                **kwargs,
            )
        self.mem.update(d, allmask)
        self.mem.save(self.out / ("memory_D%d" % stage))
        self.snapshot(checkpoint)
        save_json(
            self.out / ("D%d_update.json" % stage),
            dict(
                selected_budgets=budgets,
                selection_then_reset_then_refit=True,
                statistics_written_once=True,
            ),
        )

    def run(self):
        if (self.out / "done.json").exists():
            return
        self.d0()
        for stage in [1, 2, 3]:
            d = self.env.stage(stage)
            mask = np.ones(len(d["targets"]), bool)
            if (self.out / ("D%d_metrics.json" % stage)).exists():
                summary = json.loads((self.out / ("D%d_metrics.json" % stage)).read_text())
                for part, a in summary.items():
                    self.records.append(
                        dict(
                            dataset=self.name,
                            method=self.method,
                            seed=self.seed,
                            stage="D%d" % stage,
                            condition="clean",
                            part=part,
                            **a,
                        )
                    )
            else:
                R = self.mem.kernel(d["catalog"])
                caches = self.caches(d, self.window) if not self.no_adaptive else None
                v = evaluate(
                    self.bundle,
                    self.env,
                    d,
                    mask,
                    "clean",
                    self.out,
                    R=R,
                    li=self.li,
                    scale=self.scale,
                    window=self.window,
                    no_cold=self.no_cold,
                    no_adaptive=self.no_adaptive,
                    cache=caches,
                )
                self.publish(d, v)
            if stage < 3:
                self.update(d)
        frame = pd.DataFrame(self.records)
        for part in ["first", "follow"]:
            f = frame[frame.part == part]
            r = dict(
                dataset=self.name,
                method=self.method,
                seed=self.seed,
                stage="D1-D3",
                condition="clean",
                part=part,
                users=int(f.users.sum()),
                groups=int(f.groups.sum()),
            )
            r.update({m: float(np.average(f[m], weights=f.users)) for m in NAMES})
            self.records.append(r)
        pd.DataFrame(self.records).to_csv(self.out / "metrics.csv", index=False)
        save_json(
            self.out / "done.json",
            dict(
                completed=time.time(),
                seconds=time.time() - self.started,
                peak_GPU_bytes=torch.cuda.max_memory_allocated(),
                training_files={
                    str(p.relative_to(self.out)): sha(p)
                    for p in self.out.glob("training/*/best.pt")
                },
            ),
        )
        progress(self.out, "complete")
