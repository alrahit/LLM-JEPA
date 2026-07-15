from __future__ import annotations
import json, math
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
import torch
import torch.nn.functional as F


@torch.no_grad()
def linearity_residual(text_emb, code_emb, ridge=1e-3, normalize=True):
    if text_emb is None or code_emb is None:
        return float("nan")
    T = text_emb.detach().float(); C = code_emb.detach().float()
    if T.dim() != 2 or C.dim() != 2 or T.shape != C.shape:
        return float("nan")
    B, d = T.shape
    if B < 2:
        return float("nan")
    if normalize:
        T = F.normalize(T, dim=-1, eps=1e-8); C = F.normalize(C, dim=-1, eps=1e-8)
    I = torch.eye(d, device=T.device, dtype=T.dtype)
    A = T.t() @ T + ridge * I; Bmat = T.t() @ C
    try:
        X = torch.linalg.solve(A, Bmat)
    except RuntimeError:
        X = torch.linalg.lstsq(A, Bmat).solution
    resid = T @ X - C
    return float((resid.pow(2).sum() / (B * d)).item())


@dataclass
class LinAlignConfig:
    lambda_init: float = 1.0
    lambda_min: float = 0.0
    lambda_max: float = 16.0
    eta: float = 0.5
    tau: float = 0.05
    ema_decay: float = 0.9
    step_cap: float = 1.25
    ridge: float = 1e-3
    normalize: bool = True
    warmup_steps: int = 20
    update_every: int = 1
    min_batch: int = 16
    freeze: bool = False


class LinAlignController:
    def __init__(self, config: Optional[LinAlignConfig] = None, **kw):
        self.cfg = config or LinAlignConfig(**kw)
        self.lam = float(self.cfg.lambda_init)
        self.s_bar: Optional[float] = None
        self._n_calls = 0
        self._buf_T: List[torch.Tensor] = []
        self._buf_C: List[torch.Tensor] = []
        self._last_S = float("nan")
        self.history: List[Dict] = []

    def step(self, text_emb, code_emb) -> float:
        self._n_calls += 1
        if text_emb is not None and code_emb is not None:
            t = text_emb.detach().float(); c = code_emb.detach().float()
            if t.dim() == 2 and c.dim() == 2 and t.shape == c.shape:
                self._buf_T.append(t); self._buf_C.append(c)
        if sum(t.shape[0] for t in self._buf_T) < self.cfg.min_batch:
            return self.lam
        T = torch.cat(self._buf_T, dim=0); C = torch.cat(self._buf_C, dim=0)
        self._buf_T, self._buf_C = [], []
        S = linearity_residual(T, C, ridge=self.cfg.ridge, normalize=self.cfg.normalize)
        self._last_S = S
        valid = not (S is None or math.isnan(S) or math.isinf(S))
        if valid:
            self.s_bar = S if self.s_bar is None else (
                self.cfg.ema_decay * self.s_bar + (1 - self.cfg.ema_decay) * S)
        if (not self.cfg.freeze and valid and self.s_bar is not None
                and self._n_calls > self.cfg.warmup_steps
                and self._n_calls % self.cfg.update_every == 0):
            raw = math.exp(self.cfg.eta * (self.s_bar - self.cfg.tau))
            capped = min(max(raw, 1.0 / self.cfg.step_cap), self.cfg.step_cap)
            self.lam = float(min(max(self.lam * capped, self.cfg.lambda_min), self.cfg.lambda_max))
        return self.lam

    def log_row(self, step, extra=None):
        row = {"step": int(step), "lambda": float(self.lam), "S": float(self._last_S),
               "S_bar": (float(self.s_bar) if self.s_bar is not None else None)}
        if extra: row.update(extra)
        self.history.append(row); return row

    def print_row(self, step):
        sb = self.s_bar if self.s_bar is not None else float("nan")
        print(f"[LINALIGN] step={step} S={self._last_S:.5f} S_bar={sb:.5f} lambda={self.lam:.4f}")

    def save_history(self, path):
        with open(path, "w") as f:
            json.dump({"config": asdict(self.cfg), "history": self.history}, f, indent=2)
        print(f"[LINALIGN] history saved to {path}")


@torch.no_grad()
def verify_signal_offline(text_emb, code_emb, pca_components=50, ridge_r2=1.0):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    S = linearity_residual(text_emb, code_emb, normalize=True)
    T = text_emb.detach().float().cpu().numpy(); C = code_emb.detach().float().cpu().numpy()
    n = min(T.shape[0], C.shape[0]); k = min(pca_components, n - 1, T.shape[1])
    Tp = PCA(n_components=k).fit_transform(T[:n]); Cp = PCA(n_components=k).fit_transform(C[:n])
    Tr, Te, Cr, Ce = train_test_split(Tp, Cp, test_size=0.2, random_state=42)
    reg = Ridge(alpha=ridge_r2).fit(Tr, Cr)
    return {"S": float(S), "R2": float(r2_score(Ce, reg.predict(Te)))}


if __name__ == "__main__":
    torch.manual_seed(0); d, B = 64, 32
    T = torch.randn(B, d); W = torch.randn(d, d); C_lin = F.normalize(T, dim=-1) @ W
    c = LinAlignController(LinAlignConfig(warmup_steps=0, eta=1.0, tau=0.05, min_batch=B))
    for _ in range(10): c.step(T, C_lin)
    c.print_row(9); print(f"Linear final lambda={c.lam:.4f} (expect DOWN)")
    C_rand = torch.randn(B, d)
    c2 = LinAlignController(LinAlignConfig(warmup_steps=0, eta=1.0, tau=0.05, min_batch=B))
    for _ in range(10): c2.step(T, C_rand)
    c2.print_row(9); print(f"Random final lambda={c2.lam:.4f} (expect UP)")
    print("Grounding linear:", verify_signal_offline(T, C_lin))
    print("Grounding random:", verify_signal_offline(T, C_rand))
