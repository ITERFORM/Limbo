#!/usr/bin/env python3
# Copyright (c) 2026 ITERFORM. All rights reserved.
"""Compute layer 0's GDN output for the first token from scratch, for each candidate head pairing.

The oracle's dump prints row 0 of a 2-D tensor, which for the GDN output is head 0 — so heads 1..31 are
invisible to it and the only window is the per-layer output. This script reaches them directly: for the
first token the delta-rule state starts at zero, so the recurrence collapses to

    o[h][j] = beta[h] * (k_hat[kh] . q_hat[kh]) * v[h][j]

and the projection through ssm_out can be compared against the oracle's linear_attn_out, which it
prints. Nothing here shares code with the engine: it is a second implementation, in another language.

    python3 scripts/scan_probe.py <model.gguf> <token-id>
"""
import math
import re
import struct
import subprocess
import sys

TENSORS = ["token_embd.weight", "blk.0.attn_norm.weight", "blk.0.attn_qkv.weight",
           "blk.0.ssm_conv1d.weight", "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight",
           "blk.0.ssm_dt.bias", "blk.0.ssm_a", "blk.0.ssm_norm.weight", "blk.0.attn_gate.weight",
           "blk.0.ssm_out.weight"]


def main():
    model, token = sys.argv[1], int(sys.argv[2])

    def info(name=None):
        cmd = ["build/bin/tools/gguf_info", model] + (["--tensor", name] if name else [])
        return subprocess.run(cmd, capture_output=True, text=True).stdout

    data_off = int(re.search(r"data offset: (\d+)", info()).group(1))

    def meta(name):
        t = info(name)
        return (data_off + int(re.search(r"offset:\s+(\d+)", t).group(1)),
                int(re.search(r"nbytes:\s+(\d+)", t).group(1)))

    f = open(model, "rb")

    def read(name):
        off, n = meta(name)
        f.seek(off)
        return f.read(n)

    def q6k(buf, n):
        out, p, o = [0.0] * n, 0, 0
        while o < n:
            ql, qh = buf[p:p + 128], buf[p + 128:p + 192]
            sc = struct.unpack("<16b", buf[p + 192:p + 208])
            d = struct.unpack("<e", buf[p + 208:p + 210])[0]
            for half in range(2):
                for l in range(32):
                    i = l // 16
                    q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32
                    q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32
                    q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32
                    q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32
                    out[o + l] = d * sc[i] * q1
                    out[o + l + 32] = d * sc[i + 2] * q2
                    out[o + l + 64] = d * sc[i + 4] * q3
                    out[o + l + 96] = d * sc[i + 6] * q4
                ql, qh, sc = ql[64:], qh[32:], sc[8:]
                o += 128
            p += 210
        return out

    E, KH, VH, HD = 2048, 16, 32, 128

    def rows(name, row_elems, which):
        raw = read(name)
        rb = row_elems // 256 * 210  # Q6_K
        return [q6k(raw[r * rb:(r + 1) * rb], row_elems) for r in which]

    emb = q6k(read("token_embd.weight")[token * 1680:(token + 1) * 1680], E)
    nw = list(struct.unpack("<%df" % E, read("blk.0.attn_norm.weight")))
    inv = 1.0 / math.sqrt(sum(v * v for v in emb) / E + 1e-6)
    xn = [emb[i] * inv * nw[i] for i in range(E)]
    print("attn_norm[0..3) =", [round(v, 4) for v in xn[:3]], "(oracle -1.1480 0.5423 -1.1589)")

    raw_qkv = read("blk.0.attn_qkv.weight")
    qkv = []
    for r in range(8192):
        row = q6k(raw_qkv[r * 1680:(r + 1) * 1680], E)
        qkv.append(sum(row[i] * xn[i] for i in range(E)))
    w = list(struct.unpack("<32768f", read("blk.0.ssm_conv1d.weight")))
    conv = [qkv[ch] * w[ch * 4 + 3] / (1 + math.exp(-(qkv[ch] * w[ch * 4 + 3]))) for ch in range(8192)]

    def head_norm(vec, scale):
        out = []
        for h in range(len(vec) // HD):
            seg = vec[h * HD:(h + 1) * HD]
            s = 1.0 / math.sqrt(sum(v * v for v in seg) / HD + 1e-6) * scale
            out += [v * s for v in seg]
        return out

    q_hat = head_norm(conv[0:2048], 1.0 / math.sqrt(HD))
    k_hat = head_norm(conv[2048:4096], 1.0 / math.sqrt(HD))
    v = conv[4096:8192]

    # alpha/beta are {E, dt_rank} F32: 32 outputs each
    def proj(name):
        raw = read(name)
        # {E, 32} with ne[0]=E contiguous: element (i, o) is at i + E*o, not i*32 + o.
        return [sum(struct.unpack_from("<f", raw, (o * E + i) * 4)[0] * xn[i] for i in range(E)) for o in range(32)]

    a = proj("blk.0.ssm_alpha.weight")
    b = proj("blk.0.ssm_beta.weight")
    dtb = list(struct.unpack("<32f", read("blk.0.ssm_dt.bias")))
    sa = list(struct.unpack("<32f", read("blk.0.ssm_a")))
    print("alpha logits head", [round(v, 4) for v in a[:3]], "tail", [round(v, 4) for v in a[-3:]],
          " (oracle node_39: -4.3502 -1.6708 2.5581 ... -2.1616 -0.0000 -3.1446)")
    print("beta  logits head", [round(v, 4) for v in b[:3]], "tail", [round(v, 4) for v in b[-3:]],
          " (oracle node_45: -1.3574 -0.0012 1.7866 ... -0.4119 3.1453 -0.7332)")
    beta = [1.0 / (1.0 + math.exp(-b[h])) for h in range(VH)]
    # The decay: softplus(alpha + dt.bias) * ssm_a, and ssm_a is stored negative, so this is
    # already the log of a factor < 1.
    def softplus(x): return x if x > 20 else math.log1p(math.exp(x))
    decay = [math.exp(softplus(a[h] + dtb[h]) * sa[h]) for h in range(VH)]
    print('decay exp(gate) head', [round(v, 6) for v in decay[:3]], 'tail', [round(v, 6) for v in decay[-3:]])

    sn = list(struct.unpack("<128f", read("blk.0.ssm_norm.weight")))

    # attn_gate rows 0..4095 -> z, then silu
    raw_gate = read("blk.0.attn_gate.weight")
    z = []
    for r in range(4096):
        row = q6k(raw_gate[r * 1680:(r + 1) * 1680], E)
        z.append(sum(row[i] * xn[i] for i in range(E)))
    silu = [zz / (1 + math.exp(-zz)) for zz in z]

    # ssm_out rows 0..2 only: that is all linear_attn_out[0..3) needs
    raw_out = read("blk.0.ssm_out.weight")
    ob = 4096 // 256 * 210  # ssm_out rows are 4096 wide: 16 blocks of Q6_K
    out_rows = [q6k(raw_out[r * ob:(r + 1) * ob], 4096) for r in range(3)]

    def project(gated):
        return [sum(row[i] * gated[i] for i in range(4096)) for row in out_rows]

    combos = [("kh=h/2  beta=h", lambda h: h // 2, lambda h: h),
              ("kh=h%16 beta=h", lambda h: h % 16, lambda h: h),
              ("kh=h/2  beta=h/2", lambda h: h // 2, lambda h: h // 2),
              ("kh=h%16 beta=h/2", lambda h: h % 16, lambda h: h // 2),
              ("kh=h/2  beta=h%16", lambda h: h // 2, lambda h: h % 16),
              ("kh=h%16 beta=h%16", lambda h: h % 16, lambda h: h % 16)]
    for name, kh_of, beta_of in combos:
        o = [0.0] * (VH * HD)
        for h in range(VH):
            kh = kh_of(h)
            bh = beta_of(h)
            kq = sum(k_hat[kh * HD + i] * q_hat[kh * HD + i] for i in range(HD))
            for j in range(HD):
                o[h * HD + j] = beta[bh] * kq * v[h * HD + j]
        # variant: the same, with the decay applied to the new state's contribution as well
        o_dec = [o[i] * decay[i // HD] for i in range(VH * HD)]
        # per-head norm, times ssm_norm
        on = []
        for h in range(VH):
            seg = o[h * HD:(h + 1) * HD]
            s = 1.0 / math.sqrt(sum(t * t for t in seg) / HD + 1e-6)
            on += [seg[j] * s * sn[j] for j in range(HD)]
        gated = [on[i] * silu[i] for i in range(4096)]
        only0 = [gated[i] if i < HD else 0.0 for i in range(4096)]
        on_dec = []
        for h in range(VH):
            seg = o_dec[h * HD:(h + 1) * HD]
            sc = 1.0 / math.sqrt(sum(t * t for t in seg) / HD + 1e-6)
            on_dec += [seg[j] * sc * sn[j] for j in range(HD)]
        gated_dec = [on_dec[i] * silu[i] for i in range(4096)]
        print(f"{name}: total = {[round(t, 6) for t in project(gated)]}"
              f"  head0 only = {[round(t, 6) for t in project(only0)]}"
              f"  with decay = {[round(t, 6) for t in project(gated_dec)]}")
    print("oracle:                    linear_attn_out[0..3) = [-0.0007, 0.0005, 0.0045]")
    print("engine (map 1):            linear_attn_out[0..3) = [0.0045, 0.0043, 0.0108]")


if __name__ == "__main__":
    main()
