"""Independent market runs and saved-file comparison; no cross-market return pairing."""
from pathlib import Path
import copy
import json
import time
import pandas as pd


def compare(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = Path(folder)
    runs = json.loads((folder / 'comparison.json').read_text(encoding='utf-8'))
    frames = []
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    for market, relative in runs.items():
        run = folder / relative
        from .publication import read_daily, read_decisions
        manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
        frame = pd.read_csv(run / 'summary_by_period.csv', index_col=0)
        frame.index.name = 'policy'
        frame = frame.reset_index().assign(market=market, currency=manifest['data']['currency'],
                    return_basis=manifest['return_basis'])
        for idx, row in frame.iterrows():
            f = run / 'policies' / (row['policy'] + '_decisions.csv')
            dec = read_decisions(run, row['policy'])
            if not dec.empty:
                dec = dec.loc[row['start']:row['end']]
                for action in ['HOLD', 'TRANSFER']:
                    frame.loc[idx, action.lower()] = int((dec.action == action).sum())
        frames.append(frame)
        start, end = manifest['rows']['eval_actually_run']
        cfg = json.loads((run / 'config.json').read_text(encoding='utf-8'))
        for p in cfg['policies']:
            daily = read_daily(run, p['id'])
            daily = daily.loc[start:end]
            nav = daily.nav / daily.nav.iloc[0]
            axes[0].plot(nav.index, nav, label=f"{market}/{p['id']}")
            axes[1].plot(nav.index, nav / nav.cummax() - 1)
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(folder / 'comparison.csv', index=False)
    axes[0].set_title('Independent markets: normalized NAV (price returns, excludes dividends)')
    axes[0].legend(fontsize=7, ncol=3)
    axes[1].set_title('Drawdown; each market uses its own observed dates')
    fig.tight_layout(); fig.savefig(folder / 'comparison.png'); plt.close(fig)
    return table


def run_all(args):
    from .run import ROOT, load_config, apply_overrides, pipeline, save
    from .data import check, print_check, load_market
    if args.config or args.data:
        raise ValueError('--market all uses config/kospi200.yaml and config/sp500.yaml; no ambiguous --config/--data')
    if args.cmd == 'check-data':
        for name, file in [('kospi200', 'kospi200.yaml'), ('sp500', 'sp500.yaml')]:
            cfg = load_config(ROOT / 'config' / file)
            print_check(check(ROOT / cfg['data']['path'], cfg['data']['sha256'], market=name))
        return
    configs = {}
    for name, file in [('kospi200', 'kospi200.yaml'), ('sp500', 'sp500.yaml')]:
        cfg = apply_overrides(copy.deepcopy(load_config(ROOT / 'config' / file)), args.set)
        cfg['data']['market'] = name
        cfg['periods']['eval'] = [args.start or '2012-01-02', args.end or '2025-12-31']
        cfg['evaluation']['periods'] = {'dev': ['2012-01-02', '2019-12-31'],
                                        'test': ['2020-01-02', '2025-12-31']}
        configs[name] = cfg
    folder = ROOT / 'results' / 'comparison' / (args.run_id or time.strftime('%Y%m%d_%H%M%S'))
    folder = _reserve_folder(folder)
    print(f'Results: {folder}', flush=True)
    for name, cfg in configs.items():
        print(f'[{name}] loading data', flush=True)
        data_path = Path(cfg["data"]["path"])
        cfg["data"]["path"] = str(data_path if data_path.is_absolute() else ROOT / data_path)
        before = time.time()
        mk = load_market(cfg['data']['path'], cfg['data']['sha256'], ROOT / cfg['data']['cache_dir'], market=name)
        print(f'[{name}] running strategies and Bayesian policies', flush=True)
        save(pipeline(cfg, mk), cfg, mk, folder / name, time.time() - before)
        print(f'[{name}] completed', flush=True)
    (folder / 'comparison.json').write_text(json.dumps({n: n for n in configs}), encoding='utf-8')
    compare(folder)
    print(f'Saved {folder}')


def _reserve_folder(requested):
    """Keep previous/partial runs intact, reserving a fresh numbered sibling."""
    candidate = requested
    number = 1
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            number += 1
            candidate = requested.with_name(f"{requested.name}_{number}")
