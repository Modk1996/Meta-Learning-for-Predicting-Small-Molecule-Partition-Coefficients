#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, math, argparse, random, csv
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def r2_val(y_test, y_pred_test, y_train):
    """External R2 (Pred)"""
    y_test = np.asarray(y_test, dtype=np.float32)
    y_pred_test = np.asarray(y_pred_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_resid = y_pred_test - y_test
    SS_resid = np.sum(y_resid**2)
    y_var = y_test - np.mean(y_train)
    SS_total = np.sum(y_var**2) + 1e-12
    return float(1.0 - SS_resid / SS_total)

def pearson_r(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])

def split_indices(N, train=0.7, val=0.15, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(N); rng.shuffle(idx)
    n_tr = int(N * train); n_val = int(N * val)
    tr = idx[:n_tr]; va = idx[n_tr:n_tr + n_val]; te = idx[n_tr + n_val:]
    return tr, va, te

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_curve_csv(curve_dict: Dict[str, Dict[str, float]], csv_path: str, key_name="key"):

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([key_name, "MAE", "RMSE", "R2", "R"])
        def key2float(k: str):
            try:
                return float(k)
            except:
                return float("inf")
        for k in sorted(curve_dict.keys(), key=key2float):
            m = curve_dict[k]
            w.writerow([k, f"{m['MAE']:.6f}", f"{m['RMSE']:.6f}", f"{m['R2']:.6f}", f"{m['R']:.6f}"])


class TaskDataset:

    def __init__(self, X: np.ndarray, y: np.ndarray, name: str, seed: int = 42):
        self.name = name
        self.X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        self.y = y
        self.N = len(self.y)
        self.seed = seed
        self.rng = np.random.RandomState(seed)


        self.tr_idx, self.va_idx, self.te_idx = split_indices(self.N, seed=seed)

        y_tr = self.y[self.tr_idx]
        self.mu = float(np.mean(y_tr))
        self.sd = float(np.std(y_tr) + 1e-8)

    def _pick_indices(self, pool, shots, queries):
        total = shots + queries
        if len(pool) < total:
            queries = max(1, len(pool) - shots)
        idx = self.rng.choice(pool, size=shots + queries, replace=False)
        return idx[:shots], idx[shots:], queries

    def sample_support_query(self, split="train", shots=16, queries=64):

        pool = {"train": self.tr_idx, "val": self.va_idx, "test": self.te_idx}[split]
        s_idx, q_idx, queries = self._pick_indices(pool, shots, queries)
        Xs, ys = self.X[s_idx], self.y[s_idx]
        Xq, yq = self.X[q_idx], self.y[q_idx]
        return Xs, ys, Xq, yq


def load_systems(config_path) -> List[TaskDataset]:
    with open(config_path, "r") as f:
        cfg = json.load(f)
    tasks = []
    for s in cfg["systems"]:
        name = s["name"]; Xp = s["X"]; yp = s["y"]
        X = np.load(Xp); y = np.load(yp)
        assert len(X) == len(y), f"{name}: X/Y length mismatch"
        idx = np.arange(len(y)); np.random.shuffle(idx)
        X = X[idx]; y = y[idx]
        tasks.append(TaskDataset(X, y, name=name, seed=42))
    return tasks

def loso_partitions(tasks: List[TaskDataset], seed: int = 42):
    out = []
    K = len(tasks)
    rng = np.random.RandomState(seed)
    for i in range(K):
        test_task = tasks[i]
        remain = [tasks[j] for j in range(K) if j != i]
        rng.shuffle(remain)
        n_tr = max(1, int(0.8 * len(remain)))
        train_tasks = remain[:n_tr]
        val_tasks = remain[n_tr:] if n_tr < len(remain) else remain[-1:]
        out.append((train_tasks, val_tasks, test_task))
    return out


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(512, 256), dropout=0.1):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MAMLTrainer:
    def __init__(self, in_dim, hidden=(512, 256), dropout=0.1,
                 inner_lr=1e-3, meta_lr=1e-2, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLP(in_dim, hidden=hidden, dropout=dropout).to(self.device)
        self.inner_lr = inner_lr
        self.meta_opt = torch.optim.Adam(self.model.parameters(), lr=meta_lr)
        self.loss_fn = nn.MSELoss()

    def _clone_model(self):
        clone = MLP(self.model.net[0].in_features,
                    hidden=tuple([m.out_features for m in self.model.net if isinstance(m, nn.Linear)][:-1]),
                    dropout=0.0).to(self.device)
        clone.load_state_dict(self.model.state_dict())
        return clone

    def _inner_adapt(self, fast_model, Xs, ys, inner_steps=5):
        opt = torch.optim.SGD(fast_model.parameters(), lr=self.inner_lr)
        for _ in range(inner_steps):
            pred = fast_model(Xs)
            loss = self.loss_fn(pred, ys)
            opt.zero_grad(); loss.backward(); opt.step()

    def meta_train(self, tasks_train, tasks_val,
                   episodes=2000, shots=16, queries=64, inner_steps=5,
                   val_every=200, patience=20, seed=42, verbose=True):
        rng = np.random.RandomState(seed)
        best_val = math.inf; wait = 0; best_state = None

        for ep in range(1, episodes + 1):
            task = tasks_train[int(rng.choice(len(tasks_train), size=1)[0])]
            Xs, ys, Xq, yq = task.sample_support_query("train", shots, queries)
            Xs = torch.from_numpy(Xs).float().to(self.device)
            ys = torch.from_numpy(ys).float().to(self.device)
            Xq = torch.from_numpy(Xq).float().to(self.device)
            yq = torch.from_numpy(yq).float().to(self.device)

            fast = self._clone_model()
            self._inner_adapt(fast, Xs, ys, inner_steps)
            pred_q = fast(Xq)
            q_loss = self.loss_fn(pred_q, yq)

            grads = torch.autograd.grad(q_loss, list(fast.parameters()), retain_graph=False, create_graph=False)
            self.meta_opt.zero_grad()
            for p, g in zip(self.model.parameters(), grads):
                p.grad = g.detach().clone()
            self.meta_opt.step()

            if val_every and len(tasks_val) > 0 and ep % val_every == 0:
                val_res = self.evaluate(tasks_val, shots, queries, episodes=20, inner_steps=inner_steps)
                val_mae = val_res["MAE"]
                if verbose:
                    print(f"[MAML][ep {ep:05d}] val_MAE={val_mae:.4f}  (best={best_val:.4f}, wait={wait}/{patience})")
                if val_mae + 1e-6 < best_val:
                    best_val = val_mae; wait = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= patience and best_state is not None:
                        self.model.load_state_dict(best_state)
                        break

        if best_state is not None and val_every and len(tasks_val) > 0:
            self.model.load_state_dict(best_state)


    def _adapt_and_predict(self, task, shots, queries, inner_steps):
        Xs, ys, Xq, yq = task.sample_support_query("test", shots, queries)
        Xs_t = torch.from_numpy(Xs).float().to(self.device)
        ys_t = torch.from_numpy(ys).float().to(self.device)
        Xq_t = torch.from_numpy(Xq).float().to(self.device)

        fast = self._clone_model()
        opt = torch.optim.SGD(fast.parameters(), lr=self.inner_lr)
        for _ in range(max(1, inner_steps // 2)):
            pred = fast(Xs_t); loss = F.mse_loss(pred, ys_t)
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            yq_pred = fast(Xq_t).detach().cpu().numpy()
        return yq, yq_pred, task.y[task.tr_idx]

    def evaluate(self, tasks, shots=16, queries=64, episodes=50, inner_steps=5):
        maes, rmses, r2s, rs = [], [], [], []
        for _ in range(episodes):
            for task in tasks:
                y_true, y_pred, y_train_ref = self._adapt_and_predict(task, shots, queries, inner_steps)
                mae = float(np.mean(np.abs(y_true - y_pred)))
                rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
                r2 = r2_val(y_true, y_pred, y_train_ref)
                r = pearson_r(y_true, y_pred)
                maes.append(mae); rmses.append(rmse); r2s.append(r2); rs.append(r)
        return {"MAE": np.mean(maes), "RMSE": np.mean(rmses), "R2": np.mean(r2s), "R": np.mean(rs)}

    def evaluate_fewshot_curve(self, tasks, support_sizes=(5,10,20,30,40,50),
                               queries=64, episodes=50, inner_steps=5) -> Dict[str, Dict[str, float]]:

        out = {}
        for s in support_sizes:
            res = self.evaluate(tasks, shots=s, queries=queries, episodes=episodes, inner_steps=inner_steps)
            out[str(s)] = res
            print(f"[FewShot Curve] shots={s} → MAE={res['MAE']:.4f}, RMSE={res['RMSE']:.4f}, R2={res['R2']:.4f}, R={res['R']:.4f}")
        return out

    def evaluate_full(self, task, inner_steps=5,
                      train_ratios=(0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.8),
                      seed=42):

        rng = np.random.RandomState(seed)
        results = {}
        X_all, y_all = task.X[task.tr_idx], task.y[task.tr_idx]
        Xq, yq = task.X[task.te_idx], task.y[task.te_idx]

        for ratio in train_ratios:
            n_train = max(2, int(len(task.tr_idx) * ratio))
            idx = rng.choice(len(task.tr_idx), n_train, replace=False)
            Xs, ys = X_all[idx], y_all[idx]

            Xs_t = torch.from_numpy(Xs).float().to(self.device)
            ys_t = torch.from_numpy(ys).float().to(self.device)
            Xq_t = torch.from_numpy(Xq).float().to(self.device)

            fast = self._clone_model()
            opt = torch.optim.SGD(fast.parameters(), lr=self.inner_lr)
            for _ in range(inner_steps):
                pred = fast(Xs_t)
                loss = F.mse_loss(pred, ys_t)
                opt.zero_grad(); loss.backward(); opt.step()

            with torch.no_grad():
                yq_pred = fast(Xq_t).detach().cpu().numpy()

            mae = float(np.mean(np.abs(yq - yq_pred)))
            rmse = float(np.sqrt(np.mean((yq - yq_pred) ** 2)))
            r2 = r2_val(yq, yq_pred, ys)
            r = pearson_r(yq, yq_pred)
            results[f"{ratio:.2f}"] = {"MAE": mae, "RMSE": rmse, "R2": r2, "R": r}
            print(f"[{task.name}] TrainRatio={ratio:.2f} → MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, R={r:.4f}")

        return results

    def evaluate_clustered(self, task, train_ratio=0.1, n_clusters=None, inner_steps=5, seed=42):

        from sklearn.cluster import KMeans
        from scipy.spatial.distance import cdist

        X = task.X; y = task.y; N = len(y)
        if n_clusters is None: n_clusters = max(1, int(train_ratio * N))
        print(f"[{task.name}] (ratio={train_ratio:.2f})")

        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(X)
        centers = kmeans.cluster_centers_
        dists = cdist(centers, X)
        train_idx = np.unique(np.argmin(dists, axis=1))
        test_idx = np.array([i for i in range(N) if i not in train_idx])

        Xs, ys = X[train_idx], y[train_idx]
        Xq, yq = X[test_idx], y[test_idx]

        Xs_t = torch.from_numpy(Xs).float().to(self.device)
        ys_t = torch.from_numpy(ys).float().to(self.device)
        Xq_t = torch.from_numpy(Xq).float().to(self.device)

        fast = self._clone_model(); opt = torch.optim.SGD(fast.parameters(), lr=self.inner_lr)
        for _ in range(inner_steps):
            pred = fast(Xs_t); loss = F.mse_loss(pred, ys_t)
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            yq_pred = fast(Xq_t).detach().cpu().numpy()

        mae = float(np.mean(np.abs(yq - yq_pred)))
        rmse = float(np.sqrt(np.mean((yq - yq_pred) ** 2)))
        r2 = r2_val(yq, yq_pred, ys)
        r = pearson_r(yq, yq_pred)
        print(f"[{task.name}] ClusterEval → MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, R={r:.4f}")
        return {"MAE": mae, "RMSE": rmse, "R2": r2, "R": r}


def kfold_split(tasks: List[TaskDataset], k=3, seed=42) -> List[Tuple[List[TaskDataset], List[TaskDataset]]]:

    rng = np.random.RandomState(seed)
    idx = np.arange(len(tasks)); rng.shuffle(idx)
    folds = np.array_split(idx, k)

    splits = []
    for i in range(k):
        val_idx = folds[i]
        tr_idx = np.concatenate([folds[j] for j in range(k) if j != i]) if k > 1 else folds[i]
        train_tasks = [tasks[j] for j in tr_idx]
        val_tasks = [tasks[j] for j in val_idx]
        splits.append((train_tasks, val_tasks))
    return splits

def run_kfold_cv(in_dim, remain_tasks: List[TaskDataset], args):

    cv_splits = kfold_split(remain_tasks, k=args.cv_folds, seed=args.seed)
    cv_scores = []

    print(f"[CV] Start {args.cv_folds}-fold on {len(remain_tasks)} meta-train systems")
    for f_id, (tr_tasks, va_tasks) in enumerate(cv_splits, 1):
        trainer = MAMLTrainer(in_dim, hidden=args.hidden_tuple,
                              dropout=args.dropout, inner_lr=args.inner_lr, meta_lr=args.meta_lr)
        trainer.meta_train(tr_tasks, va_tasks,
                           episodes=args.cv_episodes,
                           shots=args.shots, queries=args.queries,
                           inner_steps=args.inner_steps,
                           val_every=max(50, args.cv_episodes // 10),
                           patience=args.patience, seed=args.seed, verbose=True)

        val_res = trainer.evaluate(va_tasks, shots=args.shots, queries=args.queries,
                                   episodes=args.eval_episodes, inner_steps=args.inner_steps)
        cv_scores.append(val_res["MAE"])
        #print(f"[CV fold {f_id}] val_MAE={val_res['MAE']:.4f}")

    best_mae = float(np.min(cv_scores))
    best_fold = int(np.argmin(cv_scores)) + 1
    #print(f"[CV] Best fold={best_fold} with val_MAE={best_mae:.4f}")
    return best_mae



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--episodes", type=int, default=2000)     
    ap.add_argument("--cv_episodes", type=int, default=1000)   
    ap.add_argument("--shots", type=int, default=50)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--eval_episodes", type=int, default=50)  
    ap.add_argument("--inner_steps", type=int, default=15)
    ap.add_argument("--hidden", type=str, default="1024,256")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--inner_lr", type=float, default=5e-3) 
    ap.add_argument("--meta_lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--cv_folds", type=int, default=3)
    ap.add_argument("--fewshot_list", type=str, default="5,10,20,30,40,50,75,100")
    ap.add_argument("--full_curve", type=str, default="0.04,0.08,0.12,0.16,0.20,0.3,0.5")
    ap.add_argument("--cluster_curve", type=str, default="0.04,0.08,0.12,0.16,0.20,0.3,0.5")
    ap.add_argument("--out_dir", type=str, default="runs_fomaml")
    args = ap.parse_args()

    ensure_dir(args.out_dir); set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    args.hidden_tuple = tuple(int(x) for x in args.hidden.split(",")) if "," in args.hidden else (int(args.hidden), 256)
    fewshot_sizes = tuple(int(x) for x in args.fewshot_list.split(","))
    full_ratios = tuple(float(x) for x in args.full_curve.split(","))
    cluster_ratios = tuple(float(x) for x in args.cluster_curve.split(","))

    save_json({
        **vars(args),
        "fewshot_sizes": fewshot_sizes,
        "full_ratios": full_ratios,
        "cluster_ratios": cluster_ratios
    }, os.path.join(args.out_dir, "run_config.json"))

    tasks = load_systems(args.config)
    in_dim = tasks[0].X.shape[1]

    folds = loso_partitions(tasks, seed=args.seed)
    print(f"\n[INFO] Starting LOSO across {len(folds)} systems ...")

    for k, (meta_train_init, meta_val_init, test_task) in enumerate(folds, 1):

        remain_tasks = meta_train_init + meta_val_init
        print(f"\n===== LOSO Fold {k}: meta-test on [{test_task.name}] =====")
        print(f"[Meta-train systems] {', '.join([t.name for t in remain_tasks])}")

        _ = run_kfold_cv(in_dim, remain_tasks, args)

        trainer = MAMLTrainer(in_dim, hidden=args.hidden_tuple,
                              dropout=args.dropout, inner_lr=args.inner_lr, meta_lr=args.meta_lr)
        trainer.meta_train(remain_tasks, tasks_val=[],   
                           episodes=args.episodes,
                           shots=args.shots, queries=args.queries,
                           inner_steps=args.inner_steps,
                           val_every=0, patience=args.patience, seed=args.seed, verbose=True)

        few_curve = trainer.evaluate_fewshot_curve([test_task],
                                                   support_sizes=fewshot_sizes,
                                                   queries=args.queries,
                                                   episodes=args.eval_episodes,
                                                   inner_steps=args.inner_steps)

        full_curve = trainer.evaluate_full(test_task, inner_steps=args.inner_steps,
                                           train_ratios=full_ratios, seed=args.seed)

        cluster_curve = {}
        for r in cluster_ratios:
            res_c = trainer.evaluate_clustered(test_task, train_ratio=r, inner_steps=args.inner_steps)
            cluster_curve[f"{r:.2f}"] = res_c

        fold_dir = os.path.join(args.out_dir, f"fold_{k}_{test_task.name}")
        ensure_dir(fold_dir)

        # Few-shot
        save_json(few_curve, os.path.join(fold_dir, "result_refit_fewshot_curve.json"))
        save_curve_csv(few_curve, os.path.join(fold_dir, "result_refit_fewshot_curve.csv"), key_name="support")

        # Full-task
        save_json(full_curve, os.path.join(fold_dir, "result_refit_full_curve.json"))
        save_curve_csv(full_curve, os.path.join(fold_dir, "result_refit_full_curve.csv"), key_name="train_ratio")

        # Clustering-based
        if cluster_curve:
            save_json(cluster_curve, os.path.join(fold_dir, "result_refit_cluster_curve.json"))
            save_curve_csv(cluster_curve, os.path.join(fold_dir, "result_refit_cluster_curve.csv"), key_name="train_ratio")

        torch.save(trainer.model.state_dict(), os.path.join(fold_dir, "model_refit.pt"))

    print("\n[DONE] All folds finished.")


if __name__ == "__main__":
    main()
