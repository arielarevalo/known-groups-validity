"""Random-direction null for the organism control (added after the primary analysis).

1,000 random unit directions per layer, oriented so that eval_eval scores above deploy_deploy
in the base model, go through the layer rule of organism_analysis.py. The statistic is the
organism minus base lambda of wood_neutral at the last prompt token.

  uv run python scripts/exploration/organism_null.py
"""
import importlib.util, numpy as np
spec = importlib.util.spec_from_file_location("oa", str(__import__("pathlib").Path(__file__).with_name("organism_analysis.py")))
oa = importlib.util.module_from_spec(spec); spec.loader.exec_module(oa)
from cv.whitebox import hua_conditions
R = 1000
blocks = oa.question_blocks(); qs = sorted(blocks)
data = {a: oa.load(a, "hua", "last_prompt") for a in oa.ARMS}
_, layers, x0 = data["organism"]; L, d = x0.shape[1], x0.shape[2]
rng = np.random.default_rng(12345)
V = rng.normal(size=(L, d, R)).astype(np.float32); V /= np.linalg.norm(V, axis=1, keepdims=True)
def cells(arm):
    ids, _, x = data[arm]
    idx = {}
    for i, rid in enumerate(ids):
        c, h, q = hua_conditions.parse_row_id(rid); idx[(c, h, q)] = i
    P = np.einsum("nld,ldr->nlr", x, V)  # rows x layers x R
    return {(c, h): P[[idx[(c, h, q)] for q in qs]] for c in ("eval_eval", "deploy_deploy", "wood_neutral") for h in (True, False)}
C = {a: cells(a) for a in oa.ARMS}
def auc(p, n):  # p,n: [q, L, R]
    s = np.concatenate([p, n]); r = s.argsort(0).argsort(0) + 1
    return (r[:len(p)].sum(0) - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))
for hint, obs in ((True, 0.866), (False, 1.237)):
    sign = np.sign(C["base"][("eval_eval", hint)].mean(0) - C["base"][("deploy_deploy", hint)].mean(0))  # orient on base
    a = {arm: auc(C[arm][("eval_eval", hint)] * sign, C[arm][("deploy_deploy", hint)] * sign) for arm in oa.ARMS}
    li = ((a["organism"] + a["base"]) / 2).argmax(0)  # per direction
    diffs = []
    for r in range(R):
        lam = {}
        for arm in oa.ARMS:
            e, dd, w = (C[arm][(c, hint)][:, li[r], r] for c in ("eval_eval", "deploy_deploy", "wood_neutral"))
            lam[arm] = (w.mean() - dd.mean()) / (e.mean() - dd.mean())
        diffs.append(lam["organism"] - lam["base"])
    diffs = np.array(diffs)
    print(f"hint {hint}: null median {np.median(diffs):+.3f}, 95th {np.percentile(diffs,95):+.3f}, share >= obs {np.mean(diffs >= obs):.3f}, mean K0 auroc {np.median((a['organism']+a['base'])[li, np.arange(R)]/2):.3f}")
