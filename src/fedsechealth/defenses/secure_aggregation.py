"""Secure aggregation with dropout recovery (Bonawitz et al., CCS 2017), simulated in-process.

Goal: the server learns the *sum* of the hospitals' updates, and nothing about
any individual update, even if some hospitals drop out mid-round.

Protocol (one round, semi-honest server; message encryption between clients
and signatures are omitted since everything runs in one process):

1. **Key agreement.** Each hospital u draws an X25519 key pair (s_u, S_u) and
   publishes S_u. Every pair (u, v) derives a shared seed
   s_uv = HKDF(DH(s_u, S_v)) without ever sending it.
2. **Secret sharing.** Each hospital also draws a self-mask seed b_u, then
   splits both s_u and b_u into n Shamir shares with threshold t and gives
   one share of each to every other hospital.
3. **Masked input.** The update x_u is encoded in fixed point in Z_{2^64} and
   sent as  y_u = x_u + PRG(b_u) + sum_{v>u} PRG(s_uv) - sum_{v<u} PRG(s_uv).
   PRG = ChaCha20 keystream. Pairwise masks cancel in the sum; the self-mask
   protects y_u if the server later falsely claims u dropped out.
4. **Unmasking.** For every surviving hospital the others reveal their
   share of b_u; for every dropped hospital they reveal their share of s_u
   (never both for the same hospital). With at least t survivors the server
   rebuilds the needed secrets, removes the self-masks and the dropped
   hospitals' dangling pairwise masks, and obtains exactly sum_{u alive} x_u.

What it does *not* protect: the aggregate itself (and the final model), and it
removes the server's ability to inspect individual updates, so it cannot be
combined naively with robust aggregation (median, Krum, ...).
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field

import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

PRIME = (
    2**521 - 1
)  # Mersenne prime; Shamir shares live in GF(PRIME), large enough for 256-bit secrets
SCALE = 2**24  # fixed-point resolution: ~6e-8, far below float32 update noise
_RAW = serialization.Encoding.Raw
_RAW_PRIV = serialization.PrivateFormat.Raw
_RAW_PUB = serialization.PublicFormat.Raw
_NO_ENC = serialization.NoEncryption()


# --------------------------------------------------------------------------- primitives


def shamir_share(secret: int, n: int, t: int) -> list[tuple[int, int]]:
    """Split ``secret`` into ``n`` shares; any ``t`` of them reconstruct it."""
    if not 1 <= t <= n:
        raise ValueError(f"threshold t={t} must be in [1, n={n}]")
    coeffs = [secret % PRIME] + [secrets.randbelow(PRIME) for _ in range(t - 1)]
    return [
        (x, sum(c * pow(x, k, PRIME) for k, c in enumerate(coeffs)) % PRIME)
        for x in range(1, n + 1)
    ]


def shamir_reconstruct(shares: list[tuple[int, int]]) -> int:
    """Lagrange interpolation at 0."""
    secret = 0
    for j, (xj, yj) in enumerate(shares):
        num, den = 1, 1
        for m, (xm, _) in enumerate(shares):
            if m != j:
                num = num * (-xm) % PRIME
                den = den * (xj - xm) % PRIME
        secret = (secret + yj * num * pow(den, -1, PRIME)) % PRIME
    return secret


def kdf(shared: bytes) -> bytes:
    return HKDF(hashes.SHA256(), 32, salt=None, info=b"fedsechealth-secagg").derive(shared)


def prg(seed: bytes, d: int) -> np.ndarray:
    """Expand a 32-byte seed into d pseudo-random uint64 values (ChaCha20 keystream)."""
    enc = Cipher(algorithms.ChaCha20(seed, bytes(16)), mode=None).encryptor()
    return np.frombuffer(enc.update(bytes(8 * d)), dtype=np.uint64).copy()


def encode(x: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(x, dtype=np.float64) * SCALE).astype(np.int64).view(np.uint64)


def decode(u: np.ndarray) -> np.ndarray:
    return u.view(np.int64).astype(np.float64) / SCALE


def _int(b: bytes) -> int:
    return int.from_bytes(b, "big")


def _bytes(i: int) -> bytes:
    return i.to_bytes(32, "big")


# --------------------------------------------------------------------------- parties


class SecAggClient:
    def __init__(self, uid: int) -> None:
        self.uid = uid
        self._sk = X25519PrivateKey.generate()
        self.pk = self._sk.public_key().public_bytes(_RAW, _RAW_PUB)
        self._b = secrets.token_bytes(32)

    def share_secrets(self, n: int, t: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Shares of (masking key, self-mask seed); share i goes to client i."""
        sk_raw = self._sk.private_bytes(_RAW, _RAW_PRIV, _NO_ENC)
        return list(
            zip(shamir_share(_int(sk_raw), n, t), shamir_share(_int(self._b), n, t), strict=True)
        )

    def masked_input(self, x: np.ndarray, public_keys: dict[int, bytes]) -> np.ndarray:
        y = encode(x) + prg(self._b, len(x))
        for v, pk in public_keys.items():
            if v == self.uid:
                continue
            m = prg(kdf(self._sk.exchange(X25519PublicKey.from_public_bytes(pk))), len(x))
            y = y + m if self.uid < v else y - m
        return y


@dataclass
class SecAggStats:
    n_clients: int
    n_dropped: int
    threshold: int
    seconds: float
    upload_bytes_per_client: int
    secrets_reconstructed: int = 0
    notes: list[str] = field(default_factory=list)


def secure_sum(
    inputs: list[np.ndarray], dropped: set[int] | None = None, threshold: int | None = None
) -> tuple[np.ndarray, SecAggStats]:
    """Run one round of the protocol; return sum of surviving inputs (float64) and stats.

    ``dropped`` hospitals complete key agreement and secret sharing, then drop
    out before sending their masked input (the hard case for the protocol).
    """
    t0 = time.perf_counter()
    n, d = len(inputs), len(inputs[0])
    dropped = set(dropped or ())
    t = threshold if threshold is not None else n // 2 + 1
    alive = [u for u in range(n) if u not in dropped]
    if len(alive) < t:
        raise RuntimeError(f"only {len(alive)} survivors, below the threshold t={t}: round aborted")

    clients = [SecAggClient(u) for u in range(n)]
    public_keys = {c.uid: c.pk for c in clients}
    # held[v][u] = (share of s_u, share of b_u) held by client v
    held: list[dict[int, tuple]] = [{} for _ in range(n)]
    for c in clients:
        for v, pair in enumerate(c.share_secrets(n, t)):
            held[v][c.uid] = pair

    masked = {u: clients[u].masked_input(inputs[u], public_keys) for u in alive}

    # Unmasking: survivors reveal b-shares for survivors, s-shares for dropped hospitals.
    responders = alive[:t]
    total = np.zeros(d, dtype=np.uint64)
    for u in alive:
        total += masked[u]
    for u in alive:
        b_u = _bytes(shamir_reconstruct([held[v][u][1] for v in responders]))
        total -= prg(b_u, d)
    for w in dropped:
        sk_w = X25519PrivateKey.from_private_bytes(
            _bytes(shamir_reconstruct([held[v][w][0] for v in responders]))
        )
        for u in alive:
            m = prg(kdf(sk_w.exchange(X25519PublicKey.from_public_bytes(public_keys[u]))), d)
            total = total - m if u < w else total + m  # undo u's dangling mask with w
    stats = SecAggStats(
        n_clients=n,
        n_dropped=len(dropped),
        threshold=t,
        seconds=time.perf_counter() - t0,
        upload_bytes_per_client=8 * d + 32 + 2 * n * 66,  # masked vector + public key + shares
        secrets_reconstructed=len(alive) + len(dropped),
    )
    return decode(total), stats
