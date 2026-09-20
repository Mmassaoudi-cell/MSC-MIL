"""Part 4: 7-way attribution on v2 (benign + SCMA-1..6). Train/val from the six cached types (train/val keys only for training), test_full for evaluation.
(a) E4 recipe with a 7-way softmax head (CE + L2 + SupCon on multi-class labels), (b) XGBoost-PF multiclass (frozen params, clean + 4 aug copies),
(c) probability fusion 0.2*tree + 0.8*net. Seeds 0-4. Resumable."""
import json, time
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
from xgboost import XGBClassifier
from src.candidates.msc_mil import MSCMILMulticlass
from src.candidates.common import supcon_loss
from src.v2 import baselines_lib as L
from src.v2 import enhanced_model as em
from src.v2.common import DATA_V2
from src.v2.robust import lib

K = 7
NAMES = ["benign"] + ["SCMA-%d" % i for i in range(1, 7)]
OUT = lib.ROB / "multiclass"
OUT.mkdir(exist_ok=True)


def build():
    """-> dict split -> D (labels 0..6), audit dict, meta."""
    zs = {t: np.load(DATA_V2 / ("scma%d_v2.npz" % t), allow_pickle=False) for t in range(1, 7)}
    splits = {}
    audit = {"note": "Benign bags are identical across the six type caches (checked); each attack class re-uses the SAME source sessions "
                     "(train: 444 sources x 9 variants per class; val/test: one attacked bag per source per class)."}
    for sp in ("train", "val", "test_full"):
        sl, mk, st, lb = [], [], [], []
        z1 = zs[1]
        ben = z1[sp + "__labels"] == 0
        ben_uids = z1[sp + "__uids"][ben]
        for t in range(1, 7):
            z = zs[t]
            zb = z[sp + "__labels"] == 0
            assert zb.sum() == ben.sum() and (z[sp + "__uids"][zb] == ben_uids).all(), "benign sets differ"
        sl.append(z1[sp + "__slots"][ben]); mk.append(z1[sp + "__mask"][ben]); st.append(z1[sp + "__static"][ben]); lb.append(np.zeros(int(ben.sum()), np.int64))
        counts = {"benign": int(ben.sum())}
        srcs = {}
        for t in range(1, 7):
            z = zs[t]
            a = z[sp + "__labels"] == 1
            sl.append(z[sp + "__slots"][a]); mk.append(z[sp + "__mask"][a]); st.append(z[sp + "__static"][a]); lb.append(np.full(int(a.sum()), t, np.int64))
            counts["SCMA-%d" % t] = int(a.sum())
            srcs[t] = set(str(u) for u in z[sp + "__x_src_uid"][a])
        d = L.D(np.concatenate(sl), np.concatenate(mk), np.concatenate(st), np.concatenate(lb))
        splits[sp] = d
        allsrc = set.union(*srcs.values())
        audit[sp] = dict(counts=counts, n_total=int(len(d.labels)), n_distinct_attack_source_sessions=len(allsrc),
                         sources_identical_across_types=all(srcs[t] == srcs[1] for t in srcs),
                         attack_sources_also_present_as_benign_bag_in_same_split=len(allsrc & set(map(str, ben_uids))))
    tr_src = set(str(u) for u in zs[1]["train__x_src_uid"][zs[1]["train__labels"] == 1])
    te_src = set(str(u) for u in zs[1]["test_full__x_src_uid"][zs[1]["test_full__labels"] == 1])
    audit["cross_split_source_overlap_train_vs_test"] = len(tr_src & te_src)
    meta = {k: zs[1]["meta__" + k] for k in ("power_mean", "power_scale", "static_mean", "static_scale")}
    for t in range(2, 7):  # the scalers are fit on the same train benign bags for every type
        for k in meta:
            assert np.allclose(meta[k], zs[t]["meta__" + k])
    return splits, audit, meta


def mc_metrics(y, prob):
    pred = prob.argmax(1)
    p, r, f, sup = precision_recall_fscore_support(y, pred, labels=list(range(K)), zero_division=0)
    cm = confusion_matrix(y, pred, labels=list(range(K)))
    ba = (y > 0).astype(int); bp = (pred > 0).astype(int)
    tp = int(((bp == 1) & (ba == 1)).sum()); fp = int(((bp == 1) & (ba == 0)).sum()); tn = int(((bp == 0) & (ba == 0)).sum()); fn = int(((bp == 0) & (ba == 1)).sum())
    return dict(macro_f1=float(f1_score(y, pred, average="macro")), accuracy=float((pred == y).mean()), precision=p.tolist(), recall=r.tolist(), f1=f.tolist(),
                support=sup.tolist(), confusion=cm.tolist(), binary_detect_recall=tp / max(tp + fn, 1), binary_far=fp / max(fp + tn, 1))


@torch.no_grad()
def net_proba(model, d, device, bs=512):
    model.eval(); out = []
    for a in range(0, len(d.labels), bs):
        out.append(model(torch.from_numpy(d.slots[a:a + bs]).to(device), torch.from_numpy(d.mask[a:a + bs]).to(device),
                         torch.from_numpy(d.static[a:a + bs]).to(device))[0].cpu().numpy())
    return np.concatenate(out)


def train_net(cfg, train, val, meta, seed, device):
    """Frozen E4 recipe (em.train_enhanced) with a 7-way head; CE instead of BCE. Early stopping on the same val score, argmax macro-F1."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = MSCMILMulticlass(3, 64, 64, 64, 4, 32, K, dropout=cfg["dropout"]).to(device)
    refs = em.noise_refs(train); fill = float(-meta["power_mean"][0] / meta["power_scale"][0])
    gen = torch.Generator(device=device).manual_seed(seed); perm_gen = torch.Generator().manual_seed(seed)
    X = torch.from_numpy(train.slots).to(device); Mk = torch.from_numpy(train.mask).to(device); S = torch.from_numpy(train.static).to(device)
    Y = torch.from_numpy(train.labels).long().to(device); N = len(Y); bs = cfg["batch_size"]
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"], eta_min=cfg["lr"] * 0.01)
    noisy_val = em.noisy_copies(val, refs, seed_offset=seed + 1000)

    def f1(d, m):
        return float(f1_score(d.labels, net_proba(m, d, device).argmax(1), average="macro"))

    best, best_state, best_ep, no_imp, converged = -1.0, None, -1, 0, True
    t0 = time.time()
    for ep in range(cfg["max_epochs"]):
        model.train(); perm = torch.randperm(N, generator=perm_gen).to(device)
        for a in range(0, N, bs):
            idx = perm[a:a + bs]
            s, m, st, y = X[idx], Mk[idx], S[idx], Y[idx]
            s, m, st = em.augment_batch(s, m, st, fill, refs[0], refs[1], cfg["noise_max"], cfg["drop_max"], gen)
            logits, _, H = model.logits_with_embedding(s, m, st)
            l2 = sum((p ** 2).sum() for p in model.parameters())
            loss = torch.nn.functional.cross_entropy(logits, y) + cfg["l2_reg"] * l2 + cfg["supcon_beta"] * supcon_loss(H, y, temperature=cfg["supcon_temperature"])
            if not torch.isfinite(loss):
                converged = False; break
            opt.zero_grad(); loss.backward(); opt.step()
        if not converged:
            break
        sched.step()
        score = 0.5 * f1(val, model) + 0.5 * float(np.mean([f1(ns, model) for ns in noisy_val]))
        if score > best:
            best, best_ep, no_imp = score, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= cfg["patience"]:
                break
        print("  ep%02d es=%.4f best=%.4f" % (ep, score, best), flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, dict(best_epoch=best_ep, best_score=best, converged=converged, train_time_s=time.time() - t0, n_params=em.n_params(model))


def fit_tree_mc(train, meta, seed):
    refs = em.noise_refs(train); fill = float(-meta["power_mean"][0] / meta["power_scale"][0])
    gen = torch.Generator().manual_seed(seed)
    S, M, ST = torch.from_numpy(train.slots), torch.from_numpy(train.mask), torch.from_numpy(train.static)
    Xs = [em.tree_features(train.slots, train.mask, train.static, meta)]
    for _ in range(em.TREE_AUG_COPIES):
        s, m, st = em.augment_batch(S, M, ST, fill, refs[0], refs[1], 0.3, 0.25, gen)
        Xs.append(em.tree_features(s.numpy(), m.numpy(), st.numpy(), meta))
    X = np.vstack(Xs); y = np.tile(train.labels, em.TREE_AUG_COPIES + 1)
    est = XGBClassifier(eval_metric="mlogloss", objective="multi:softprob", num_class=K, random_state=seed, n_jobs=8, tree_method="hist", **em.FROZEN_TREE_PARAMS)
    est.fit(X, y)
    return est


def main():
    device = lib.dev()
    sp, audit, meta = build()
    (lib.ROB / "multiclass_data_audit.json").write_text(json.dumps(audit, indent=1))
    print(json.dumps(audit, indent=1), flush=True)
    tr, va, te = sp["train"], sp["val"], sp["test_full"]
    np.save(OUT / "labels_test.npy", te.labels)
    for s in lib.SEEDS5:
        out = OUT / ("multiclass_seed%d.json" % s)
        if out.exists():
            continue
        t0 = time.time()
        net, info = train_net(em.FROZEN_NET_CFG, tr, va, meta, s, device)
        pn = net_proba(net, te, device)
        tree = fit_tree_mc(tr, meta, s)
        pt = tree.predict_proba(em.tree_features(te.slots, te.mask, te.static, meta))
        pf = em.FROZEN_W_TREE * pt + (1 - em.FROZEN_W_TREE) * pn
        for nm, p in (("net", pn), ("tree", pt), ("fusion", pf)):
            np.save(OUT / ("prob_%s_seed%d.npy" % (nm, s)), p.astype(np.float16))
        res = dict(seed=s, net_info=info, w_tree=em.FROZEN_W_TREE, net=mc_metrics(te.labels, pn), tree=mc_metrics(te.labels, pt), fusion=mc_metrics(te.labels, pf))
        out.write_text(json.dumps(res, default=float))
        print("multiclass seed%d %.0fs macroF1 net=%.4f tree=%.4f fusion=%.4f" % (s, time.time() - t0, res["net"]["macro_f1"], res["tree"]["macro_f1"], res["fusion"]["macro_f1"]), flush=True)


if __name__ == "__main__":
    main()
