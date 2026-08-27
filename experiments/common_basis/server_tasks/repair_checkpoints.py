"""Repair corrupted SB3 checkpoints (policy intact, optimizer.pth corrupt).

Builds a valid but empty optimizer state from the intact checkpoint, then
rebuilds each corrupted zip: original data/policy/pytorch_variables +
placeholder optimizer. Works because measurement tasks never touch the
optimizer state values.
"""
import io
import os
import sys
import zipfile

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import json_to_data

SRC = r"C:\Users\Flavi\opencode\IIGC\data\models_overcooked"
DST = r"C:\Users\Flavi\AppData\Local\Temp\opencode\chkpt_clean"
INTACT = os.path.join(SRC, "overcookedv3_dynamic_seed41_final.zip")


def make_fake_optim():
    model = PPO.load(INTACT, device="cpu")
    sd = model.policy.optimizer.state_dict()
    sd["state"] = {}
    return sd


def rebuild(fp, optim_sd, out):
    with zipfile.ZipFile(fp) as zin:
        data = zin.read("data")
        policy = zin.read("policy.pth")
        pv = zin.read("pytorch_variables.pth")
    with zipfile.ZipFile(out, "w") as zout:
        zout.writestr("data", data)
        zout.writestr("pytorch_variables.pth", pv)
        zout.writestr("policy.pth", policy)
        buf = io.BytesIO()
        torch.save(optim_sd, buf)
        zout.writestr("policy.optimizer.pth", buf.getvalue())
        zout.writestr("_stable_baselines3_version", "2.8.0")


def usable(fp):
    z = zipfile.ZipFile(fp)
    ok = {}
    for n in ("data", "policy.pth", "pytorch_variables.pth"):
        try:
            z.read(n)
            ok[n] = True
        except Exception:
            ok[n] = False
    z.close()
    return ok


def main():
    os.makedirs(DST, exist_ok=True)
    optim_sd = make_fake_optim()
    fixed, skipped = [], []
    for name in sorted(os.listdir(SRC)):
        if not name.endswith(".zip"):
            continue
        fp = os.path.join(SRC, name)
        if not zipfile.is_zipfile(fp):
            print("NOTAZIP ", name)
            continue
        ok = usable(fp)
        if all(ok.values()) and zipfile.ZipFile(fp).testzip() is None:
            print("intact  ", name)
            continue
        if not ok["policy.pth"]:
            skipped.append(name)
            print("UNUSABLE", name)
            continue
        rebuild(fp, optim_sd, os.path.join(DST, name))
        fixed.append(name)
        print("rebuilt ", name)
    print(f"\nfixed={len(fixed)} unusable={len(skipped)} -> {DST}")
    if skipped:
        print("skipped:", skipped)
    for name in fixed:
        p = os.path.join(DST, name)
        try:
            PPO.load(p, device="cpu")
            print("VERIFY OK  ", name)
        except Exception as e:
            print("VERIFY FAIL", name, "->", type(e).__name__, e)


if __name__ == "__main__":
    main()
