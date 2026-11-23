"""Compact, lossless report inputs; source data is never part of the publication."""
from pathlib import Path
from contextlib import contextmanager
import tempfile
import json
import shutil
import pandas as pd


def _read(folder, stem, suffix, policy):
    folder = Path(folder)
    packed = folder / (stem + ".csv.gz")
    if packed.exists():
        df = pd.read_csv(packed, float_precision="round_trip", low_memory=False)
        df = df.loc[df.policy == policy].drop(columns="policy")
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")
    f = folder / "policies" / (policy + suffix)
    return pd.read_csv(f, index_col=0, parse_dates=True, float_precision="round_trip", low_memory=False) if f.exists() else pd.DataFrame()


def read_daily(folder, policy):
    return _read(folder, "daily", "_daily.csv", policy)


def read_decisions(folder, policy):
    return _read(folder, "decisions", "_decisions.csv", policy)


def pack(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    schemas = {}
    for stem, suffix in [("daily", "_daily.csv"), ("decisions", "_decisions.csv"), ("forecasts", "_forecasts.csv")]:
        parts = []
        schemas[stem] = {}
        for file in sorted((source / "policies").glob("*" + suffix)):
            df = pd.read_csv(file, float_precision="round_trip", low_memory=False)
            schemas[stem][file.name.removesuffix(suffix)] = list(df.columns)
            df.insert(0, "policy", file.name.removesuffix(suffix))
            parts.append(df)
        pd.concat(parts, ignore_index=True).to_csv(destination / (stem + ".csv.gz"), index=False,
                                                    compression={"method": "gzip", "mtime": 0})
    (destination / "schema.json").write_text(json.dumps(schemas), encoding="utf-8")
    pd.read_csv(source / "policies/sleeves.csv", float_precision="round_trip", low_memory=False).to_csv(
        destination / "allocation.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    for file in source.glob("*.csv"):
        shutil.copyfile(file, destination / file.name)
    for name in ["manifest.json", "config.json"]:
        obj = json.loads((source / name).read_text(encoding="utf-8"))
        data = obj.get("data", {})
        if "path" in data:
            data["path"] = "data/raw/" + Path(data["path"]).name
        obj["publication_path_normalized"] = True
        (destination / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copytree(source / "figures", destination / "figures")


@contextmanager
def unpacked(folder):
    folder = Path(folder)
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary)
        (target / "policies").mkdir()
        for name in ["config.json", "manifest.json"]:
            shutil.copyfile(folder / name, target / name)
        schemas = json.loads((folder / "schema.json").read_text(encoding="utf-8"))
        for stem, suffix in [("daily", "_daily.csv"), ("decisions", "_decisions.csv"), ("forecasts", "_forecasts.csv")]:
            df = pd.read_csv(folder / (stem + ".csv.gz"), float_precision="round_trip", low_memory=False)
            for policy, part in df.groupby("policy", sort=False):
                if not isinstance(policy, str) or not policy.replace("_", "").isalnum():
                    raise ValueError("Invalid policy identifier")
                part[schemas[stem][policy]].to_csv(target / "policies" / (policy + suffix), index=False)
        pd.read_csv(folder / "allocation.csv.gz", float_precision="round_trip", low_memory=False).to_csv(target / "policies/sleeves.csv", index=False)
        yield target
