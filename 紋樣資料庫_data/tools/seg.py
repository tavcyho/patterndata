#!/usr/bin/env python3
"""Pattern-plate segmentation: render a PDF page, find the individual motifs on it (XY-cut on an ink mask),
return bounding boxes.  Used by build.py."""
import subprocess, os, sys, tempfile
import numpy as np, cv2
from scipy import ndimage as ndi


def render_page(pdf, page, long_edge, tmpdir):
    base = os.path.join(tmpdir, "pg")
    subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-scale-to", str(long_edge), "-png", pdf, base],
                   check=True, capture_output=True)
    fn = [f for f in os.listdir(tmpdir) if f.startswith("pg") and f.endswith(".png")][0]
    p = os.path.join(tmpdir, fn)
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    os.remove(p)
    return im


def _pct_bg(g, pct=85, win_frac=0.10, ds=8):
    small = cv2.resize(g, (max(2, g.shape[1] // ds), max(2, g.shape[0] // ds)), interpolation=cv2.INTER_AREA)
    w = max(3, int(max(small.shape) * win_frac) | 1)
    bg = ndi.percentile_filter(small, pct, size=w, mode="nearest")
    bg = cv2.GaussianBlur(bg, (0, 0), w / 4)
    return cv2.resize(bg, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def deskew(bgr, max_angle=2.0, step=0.1):
    """Rotate page so that text/ornament rows are horizontal (projection-profile method)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    ds = max(1, max(H, W) // 900)
    sm = cv2.resize(gray, (W // ds, H // ds), interpolation=cv2.INTER_AREA).astype(np.float32)
    dark_panel = (sm < 70).mean() > 0.30
    sm = 255 - sm if not dark_panel else sm
    bg = np.percentile(sm, 20)
    inkf = np.clip(sm - bg, 0, None)
    inkf = (inkf > np.percentile(inkf, 80) * 0.5).astype(np.float32)
    h, w = inkf.shape
    m = int(0.05 * max(h, w))
    inkf = inkf[m:h - m, m:w - m]
    hh, ww = inkf.shape
    best, ba = -1, 0.0
    for a in np.arange(-max_angle, max_angle + 1e-6, step):
        M = cv2.getRotationMatrix2D((ww / 2, hh / 2), a, 1.0)
        r = cv2.warpAffine(inkf, M, (ww, hh), flags=cv2.INTER_LINEAR)
        p = r.sum(1)
        sc = np.sum(np.diff(p) ** 2)
        if sc > best:
            best, ba = sc, a
    if abs(ba) < 0.15:
        return bgr, 0.0
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ba, 1.0)
    border = tuple(int(v) for v in np.median(bgr.reshape(-1, 3), axis=0))
    out = cv2.warpAffine(bgr, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    return out, float(ba)


def find_roi(gray, margin_frac):
    """Return (x0,y0,x1,y1) of the working area.  If a big dark panel exists (black plate) use it."""
    H, W = gray.shape
    L = max(H, W)
    m = int(margin_frac * L)
    dark = (gray < 70).astype(np.uint8)
    small0 = cv2.resize(dark, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    small0 = (small0 > 0.5).astype(np.uint8)
    small = cv2.morphologyEx(small0, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(small, 8)
    if n > 1:
        k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        x, y, w, h = [int(v) for v in stats[k, :4]]
        fillfrac = float(small0[y:y + h, x:x + w].mean())
        if w * h * 64 > 0.20 * H * W and fillfrac > 0.22:
            return (x * 8 + m // 2, y * 8 + m // 2, (x + w) * 8 - m // 2, (y + h) * 8 - m // 2), True
    # paper bounding box (drop scanner-bed / shadow borders): columns/rows whose 80th-percentile brightness is paper-like
    sm = cv2.resize(gray, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    ref = np.percentile(sm, 85)
    colp = np.percentile(sm, 80, axis=0)
    rowp = np.percentile(sm, 80, axis=1)
    cx = np.where(colp > 0.55 * ref)[0]
    ry = np.where(rowp > 0.55 * ref)[0]
    if len(cx) > 10 and len(ry) > 10:
        return (int(cx[0]) * 4 + m, int(ry[0]) * 4 + m, (int(cx[-1]) + 1) * 4 - m, (int(ry[-1]) + 1) * 4 - m), False
    return (m, m, W - m, H - m), False


cfg_dark_thr = 30
cfg_dark_delta = 26


def ink_mask(bgr, roi, dark_panel, thr=0.80, mode="ink"):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    x0, y0, x1, y1 = roi
    g = gray.astype(np.float32)
    if dark_panel and mode == "ink":
        gsm = cv2.GaussianBlur(gray, (0, 0), 1.5).astype(np.float32)
        small = cv2.resize(gsm, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
        lo = ndi.percentile_filter(small, 10, size=15, mode="nearest")
        lo = cv2.resize(cv2.GaussianBlur(lo, (0, 0), 2), (W, H), interpolation=cv2.INTER_LINEAR)
        ink = ((gsm > lo + cfg_dark_delta) & (gsm > cfg_dark_thr)).astype(np.uint8)
        nn, lb, st, _ = cv2.connectedComponentsWithStats(ink, 8)
        mx = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT])
        keepc = mx >= 0.018 * max(H, W)
        keepc[0] = False
        ink = keepc[lb].astype(np.uint8)
        m = np.zeros_like(ink); m[y0:y1, x0:x1] = 1
        return ink & m
    if mode == "edges":
        # texture / edge map for photos of plaster on a plain backdrop
        gs = cv2.GaussianBlur(gray, (0, 0), max(1.0, 0.0008 * max(H, W)))
        gx = cv2.Sobel(gs, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gs, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 0.002 * max(H, W))
        sub = mag[y0:y1, x0:x1]
        t = max(np.percentile(sub, 60) * 1.0, 3.0)
        ink = (mag > t)
    else:
        bg = _pct_bg(g)
        ink = (g / np.maximum(bg, 1) < thr)
    m = np.zeros_like(ink)
    m[y0:y1, x0:x1] = True
    ink &= m
    return ink.astype(np.uint8)


def despeckle(ink, min_area):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab].astype(np.uint8)


def _runs(empty):
    """yield (start, length) of True runs strictly inside the array (not touching either end)."""
    n = len(empty)
    i = 0
    out = []
    while i < n:
        if empty[i]:
            j = i
            while j < n and empty[j]:
                j += 1
            if i > 0 and j < n:
                out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


def xycut(ink, L, params):
    H, W = ink.shape
    leaves = []

    def tight(box):
        x0, y0, x1, y1 = box
        sub = ink[y0:y1, x0:x1]
        cols = np.where(sub.sum(0) > 0)[0]
        rows = np.where(sub.sum(1) > 0)[0]
        if len(cols) == 0:
            return None
        return (x0 + cols[0], y0 + rows[0], x0 + cols[-1] + 1, y0 + rows[-1] + 1)

    def rec(box, depth=0):
        box = tight(box)
        if box is None:
            return
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        sub = ink[y0:y1, x0:x1]
        eps_r = params["eps"] * w
        eps_c = params["eps"] * h
        rp = sub.sum(1) <= eps_r
        cp = sub.sum(0) <= eps_c
        best = None
        gh_min = max(params["gh_abs"] * L, params["gh_rel"] * h)
        for s, ln in _runs(rp):
            r = ln / gh_min
            if ln >= gh_min and (best is None or r > best[0]):
                best = (r, "h", s, ln)
        gv_min = max(params["gv_abs"] * L, min(params["gv_rel"] * h, params["gv_cap"] * L))
        for s, ln in _runs(cp):
            r = ln / gv_min
            if ln >= gv_min and (best is None or r > best[0]):
                best = (r, "v", s, ln)
        if best is None or depth > 40:
            leaves.append(box)
            return
        _, d, s, ln = best
        if d == "h":
            rec((x0, y0, x1, y0 + s + ln // 2), depth + 1)
            rec((x0, y0 + s + ln // 2, x1, y1), depth + 1)
        else:
            rec((x0, y0, x0 + s + ln // 2, y1), depth + 1)
            rec((x0 + s + ln // 2, y0, x1, y1), depth + 1)

    rec((0, 0, W, H))
    return leaves


DEFAULT = dict(eps=0.0015, gh_abs=0.0035, gh_rel=0.008, gv_abs=0.008, gv_rel=0.30, gv_cap=0.013)


def segment(bgr, cfg):
    if cfg.get("deskew", True):
        bgr, ang = deskew(bgr)
    else:
        ang = 0.0
    H, W = bgr.shape[:2]
    L = max(H, W)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    roi, dark = find_roi(gray, cfg.get("margin", 0.025))
    if cfg.get("roi_l") or cfg.get("roi_r"):
        roi = (max(roi[0], int(cfg.get("roi_l", 0) * W)), roi[1], min(roi[2], int(cfg.get("roi_r", 1.0) * W)), roi[3])
    is_photo = False
    if cfg.get("auto") and not dark:
        # photo page = a big leaf whose interior is clearly darker than the paper (grey backdrop)
        gbl = cv2.GaussianBlur(gray, (0, 0), 4)
        sub = gbl[roi[1]:roi[3], roi[0]:roi[2]]
        paper = float(np.percentile(sub, 95))
        pcfg = dict(cfg); pcfg["thr"] = 0.93; pcfg["auto"] = False
        pres, _, _, _, _ = segment(bgr, dict(pcfg, deskew=False))
        for r in pres:
            x0, y0, x1, y1 = r["box"]
            if (x1 - x0) * (y1 - y0) > 0.25 * (roi[2] - roi[0]) * (roi[3] - roi[1]):
                med = float(np.median(gbl[y0:y1, x0:x1]))
                if med < paper - cfg.get("photo_gap", 22):
                    is_photo = True
                    photo_leaf = r
                    break
        if is_photo:
            cfg = dict(cfg); cfg["thr"] = 0.93
    ink = ink_mask(bgr, roi, dark, thr=cfg.get("thr", 0.80), mode=cfg.get("mode", "ink"))
    ink = despeckle(ink, max(6, int(cfg.get("speck", 0.00003) * H * W)))
    ink = strip_dark_edges(ink, roi, cfg)
    # bridge hairlines / dotted lines a bit so they are not separate pieces
    k = max(1, int(cfg.get("bridge", 0.0015) * L))
    cut_ink = cv2.dilate(ink, np.ones((k, k), np.uint8)) if k > 1 else ink
    params = dict(DEFAULT)
    params.update(cfg.get("cut", {}))
    boxes = xycut(cut_ink, L, params)
    def mk(x0, y0, x1, y1):
        sub = ink[y0:y1, x0:x1]
        cols = np.where(sub.sum(0) > 0)[0]
        rows = np.where(sub.sum(1) > 0)[0]
        if len(cols) == 0:
            return None
        x0, x1 = x0 + cols[0], x0 + cols[-1] + 1
        y0, y1 = y0 + rows[0], y0 + rows[-1] + 1
        w, h = x1 - x0, y1 - y0
        dens = float(ink[y0:y1, x0:x1].mean())
        # ---- filters
        if max(w, h) < cfg.get("min_long", 0.035) * L or min(w, h) < cfg.get("min_short", 0.012) * L:
            return None
        if h < cfg.get("text_h", 0.024) * L and w / h > 1.8:
            return None                   # single line of caption / price text
        if w < cfg.get("vtext_w", 0.04) * L and h / w > 3 and dens < 0.30:
            return None                   # vertical caption (one or two lines)
        if w < cfg.get("text_h", 0.024) * L and h / w > 1.8:
            return None
        et = cfg.get("edge_text")
        if et:
            rx0, ry0, rx1, ry1 = roi
            rw = rx1 - rx0
            if (x1 < rx0 + et * rw or x0 > rx1 - et * rw) and w < 0.06 * L and dens < 0.35:
                return None               # marginal running caption
        if cfg.get("text_filter", True) and max(w, h) < 0.45 * L and looks_like_text(ink[y0:y1, x0:x1]):
            return None
        ar = max(w / h, h / w)
        if ar > 9 and dens > 0.45 and min(w, h) < 0.03 * L:
            return None                   # scan-edge shadow / gutter
        return dict(box=(int(x0), int(y0), int(x1), int(y1)), density=round(dens, 3))

    res = []
    for (x0, y0, x1, y1) in boxes:
        r = mk(x0, y0, x1, y1)
        if r:
            res.append(r)
    if cfg.get("collage_split", True) and not is_photo:
        # stacked strips that the first cut could not separate (skew, tight gaps): retry with a finer cut
        relaxed = dict(DEFAULT); relaxed.update(dict(eps=0.004, gh_abs=0.0016, gh_rel=0.004))
        relaxed.update(cfg.get("collage_cut", {}))
        new = []
        for r in res:
            x0, y0, x1, y1 = r["box"]
            if (y1 - y0) >= 0.16 * H and (x1 - x0) >= 0.15 * W:
                sub_boxes = xycut(cut_ink[y0:y1, x0:x1], L, relaxed)
                pieces = []
                for (a0, b0, a1, b1) in sub_boxes:
                    q = mk(x0 + a0, y0 + b0, x0 + a1, y0 + b1)
                    if q:
                        pieces.append(q)
                if len(pieces) >= 3:
                    new.extend(pieces); continue
            new.append(r)
        res = new
    if cfg.get("big_split", True) and not is_photo:
        new = []
        for r in res:
            x0, y0, x1, y1 = r["box"]
            if (x1 - x0) * (y1 - y0) > cfg.get("big_frac", 0.14) * H * W:
                sp = split_big_leaf(ink, r["box"], L, H, W, cfg)
                if sp:
                    new.extend(sp); continue
            new.append(r)
        res = new
    if is_photo:
        objs = photo_split(bgr, roi, dict(cfg, edge_text=cfg.get("edge_text", 0.0)))
        if len(objs) >= cfg.get("photo_objs_min", 4):
            res = objs
        else:
            # single big photo / moulding plate: keep the largest leaf only
            res = [photo_leaf]
    if cfg.get("trim_panel"):
        res = trim_panels(gray, res, cfg, L)
    if cfg.get("photo_split"):
        res = photo_split(bgr, roi, cfg)
    elif cfg.get("panel_only"):
        res = panel_only(bgr, roi, cfg)
    if is_photo:
        ang = (ang, 'photo')
    return res, roi, dark, bgr, ang


def find_panel(gray, roi, off=10):
    """Bounding box of the (grey) photo panel printed on white paper, inside roi."""
    x0, y0, x1, y1 = roi
    ds = 8
    sub = cv2.GaussianBlur(gray[y0:y1, x0:x1], (0, 0), 5)
    small = cv2.resize(sub, (sub.shape[1] // ds, sub.shape[0] // ds), interpolation=cv2.INTER_AREA)
    k = max(5, int(0.5 * max(small.shape))) | 1
    paper = cv2.GaussianBlur(cv2.dilate(small, np.ones((k, k), np.uint8)), (0, 0), k / 6)
    m = (small.astype(np.float32) < paper.astype(np.float32) - off).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((13, 13), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None
    kk = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, w, h = [int(v) * ds for v in stats[kk, :4]]
    return (x0 + x, y0 + y, x0 + x + w, y0 + y + h)


def trim_panels(gray, res, cfg, L):
    """Shrink each item to the grey/dark photo panel inside it (drops printed captions & shading strips)."""
    out = []
    for r in res:
        x0, y0, x1, y1 = r["box"]
        sub = gray[y0:y1, x0:x1]
        cm = cv2.GaussianBlur(np.median(sub, axis=0).astype(np.float32).reshape(1, -1), (0, 0), 3).reshape(-1)
        rm = cv2.GaussianBlur(np.median(sub, axis=1).astype(np.float32).reshape(1, -1), (0, 0), 3).reshape(-1)
        paper = np.percentile(cm, 95)
        def longest(v):
            ok = v < paper - cfg.get("trim_off", 14)
            best = (0, 0); i = 0; n = len(ok)
            while i < n:
                if ok[i]:
                    j = i
                    gap = 0
                    k = i
                    while k < n and gap < 0.01 * L:
                        if ok[k]:
                            j = k; gap = 0
                        else:
                            gap += 1
                        k += 1
                    if j - i > best[1] - best[0]:
                        best = (i, j + 1)
                    i = j + 1
                else:
                    i += 1
            return best
        cx0, cx1 = longest(cm)
        cy0, cy1 = longest(rm)
        if cx1 - cx0 < 0.1 * L or cy1 - cy0 < 0.1 * L:
            out.append(r); continue
        out.append(dict(box=(x0 + cx0, y0 + cy0, x0 + cx1, y0 + cy1), density=r["density"]))
    return out


def photo_split(bgr, roi, cfg):
    """Separate the individual objects lying on the plain backdrop of a photo panel."""
    H, W = bgr.shape[:2]
    L = max(H, W)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    x0, y0, x1, y1 = roi
    pad = int(cfg.get("panel_pad", 0.01) * L)
    px0, py0, px1, py1 = x0 + pad, y0 + pad, x1 - pad, y1 - pad
    g = gray[py0:py1, px0:px1].astype(np.float32)
    gb = cv2.GaussianBlur(g, (0, 0), max(1.5, 0.0012 * L))
    bg = _pct_bg(gb, pct=50, win_frac=0.35)
    dev = np.abs(gb - bg)
    ink = (dev > cfg.get("photo_dev", 20)).astype(np.uint8)
    ink = despeckle(ink, max(20, int(0.00005 * H * W)))
    k = max(3, int(cfg.get("photo_close", 0.006) * L)) | 1
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    ink = ndi.binary_fill_holes(ink).astype(np.uint8)
    ink = despeckle(ink, max(80, int(0.0002 * H * W)))
    # objects are not guillotine-separable (interleaved rosettes etc.) -> connected components;
    # small fragments (numbers, detached leaves) are attached to the nearest big object.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    comps = []
    for i in range(1, n):
        x, y, w, h, a = [int(v) for v in stats[i]]
        comps.append([x, y, x + w, y + h, a])
    big_a = cfg.get("photo_big", 0.0012) * H * W
    bigs = [c for c in comps if c[4] >= big_a]
    smalls = [c for c in comps if c[4] < big_a]
    att = int(cfg.get("photo_attach", 0.02) * L)
    keep = []
    for c in smalls:
        best, bd = None, 1e9
        for b in bigs:
            dx = max(b[0] - c[2], c[0] - b[2], 0); dy = max(b[1] - c[3], c[1] - b[3], 0)
            d = max(dx, dy)
            if d < bd:
                best, bd = b, d
        if best is not None and bd <= att:
            best[0], best[1], best[2], best[3] = min(best[0], c[0]), min(best[1], c[1]), max(best[2], c[2]), max(best[3], c[3])
        elif c[4] >= 0.0004 * H * W:
            keep.append(c)
    boxes = bigs + keep
    et = cfg.get("edge_text")
    sub = []
    for (a0, b0, a1, b1, ar) in boxes:
        w, h = a1 - a0, b1 - b0
        if max(w, h) < cfg.get("photo_min", 0.05) * L or min(w, h) < 0.02 * L:
            continue
        if max(w / h, h / w) > 4 and min(w, h) < 0.035 * L:
            continue
        if et:
            rx0, ry0, rx1, ry1 = roi
            if (px0 + a1) < rx0 + et * (rx1 - rx0) or (px0 + a0) > rx1 - et * (rx1 - rx0):
                continue
        sub.append(dict(box=(px0 + a0, py0 + b0, px0 + a1, py0 + b1), density=round(float(ar) / (w * h), 3)))
    if cfg.get("photo_union") and sub:
        # one item = everything that is not a marginal caption (single big photo / moulding plate)
        big = [r for r in sub if (r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1]) > 0.004 * H * W]
        if big:
            sub = [dict(box=(min(r["box"][0] for r in big), min(r["box"][1] for r in big),
                             max(r["box"][2] for r in big), max(r["box"][3] for r in big)), density=1.0)]
    return sub


def panel_only(bgr, roi, cfg):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    p = find_panel(gray, roi)
    if p is None:
        return []
    return [dict(box=tuple(int(v) for v in p), density=1.0)]


def strip_dark_edges(ink, roi, cfg):
    """Remove scan shadows: columns/rows hugging the ROI border that are (almost) solid ink."""
    if cfg.get("keep_edges"):
        return ink
    x0, y0, x1, y1 = roi
    w, h = x1 - x0, y1 - y0
    thr = cfg.get("edge_solid", 0.45)
    colf = ink[y0:y1, x0:x1].mean(axis=0)
    rowf = ink[y0:y1, x0:x1].mean(axis=1)
    lim_c, lim_r = int(0.10 * w), int(0.06 * h)
    for side in ("l", "r"):
        rng = range(lim_c) if side == "l" else range(w - 1, w - 1 - lim_c, -1)
        last = -1
        for i, c in enumerate(rng):
            if colf[c] > thr:
                last = i
        if last >= 0:
            cols = list(rng)[:last + 1]
            for c in cols:
                ink[y0:y1, x0 + c] = 0
    for side in ("t", "b"):
        rng = range(lim_r) if side == "t" else range(h - 1, h - 1 - lim_r, -1)
        last = -1
        for i, r in enumerate(rng):
            if rowf[r] > thr:
                last = i
        if last >= 0:
            for r in list(rng)[:last + 1]:
                ink[y0 + r, x0:x1] = 0
    return ink


def looks_like_text(ink_leaf):
    """small aligned glyph-like components arranged in >=2 rows (price lists, captions)"""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink_leaf, 8)
    comps = [stats[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 12]
    if len(comps) < 8:
        return False
    hs = np.array([c[cv2.CC_STAT_HEIGHT] for c in comps], float)
    ws = np.array([c[cv2.CC_STAT_WIDTH] for c in comps], float)
    lh = ink_leaf.shape[0]
    mh = np.median(hs)
    if mh > 0.34 * lh or np.std(hs) / mh > 0.45:
        return False
    cy = np.array([c[cv2.CC_STAT_TOP] + c[cv2.CC_STAT_HEIGHT] / 2 for c in comps])
    order = np.argsort(cy)
    rows, cur = [], [order[0]]
    for i in order[1:]:
        if cy[i] - cy[cur[-1]] <= 0.6 * mh:
            cur.append(i)
        else:
            rows.append(cur); cur = [i]
    rows.append(cur)
    big_rows = [r for r in rows if len(r) >= 3]
    return len(big_rows) >= 2 or (len(big_rows) >= 1 and mh < 0.09 * lh and len(comps) >= 10)


def cc_boxes(mask, big_a, attach_px, keep_a):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    comps = []
    for i in range(1, n):
        x, y, w, h, a = [int(v) for v in stats[i]]
        comps.append([x, y, x + w, y + h, a])
    bigs = [c for c in comps if c[4] >= big_a]
    keep = []
    for c in [c for c in comps if c[4] < big_a]:
        best, bd = None, 1e9
        for b in bigs:
            d = max(max(b[0] - c[2], c[0] - b[2], 0), max(b[1] - c[3], c[1] - b[3], 0))
            if d < bd:
                best, bd = b, d
        if best is not None and bd <= attach_px:
            best[0], best[1], best[2], best[3] = min(best[0], c[0]), min(best[1], c[1]), max(best[2], c[2]), max(best[3], c[3])
        elif c[4] >= keep_a:
            keep.append(c)
    return bigs + keep


def split_big_leaf(ink, box, L, H, W, cfg):
    """A leaf that is still huge and made of interleaved shapes: split it by connected components."""
    x0, y0, x1, y1 = box
    sub = ink[y0:y1, x0:x1]
    k = max(3, int(cfg.get("big_close", 0.008) * L)) | 1
    m = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = ndi.binary_fill_holes(m).astype(np.uint8)
    la = (x1 - x0) * (y1 - y0)
    boxes = cc_boxes(m, 0.015 * la, int(0.02 * L), 0.004 * la)
    boxes = [b for b in boxes if max(b[2] - b[0], b[3] - b[1]) >= 0.04 * L]
    if len(boxes) < 3:
        return None
    out = []
    for (a0, b0, a1, b1, ar) in boxes:
        out.append(dict(box=(x0 + a0, y0 + b0, x0 + a1, y0 + b1), density=round(float(ink[y0 + b0:y0 + b1, x0 + a0:x0 + a1].mean()), 3)))
    return out


def draw_debug(bgr, res, out, target=1100):
    H, W = bgr.shape[:2]
    s = target / max(H, W)
    im = cv2.resize(bgr, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    for i, r in enumerate(res):
        x0, y0, x1, y1 = [int(v * s) for v in r["box"]]
        col = (0, 0, 255) if i % 2 == 0 else (255, 0, 0)
        cv2.rectangle(im, (x0, y0), (x1, y1), col, 2)
        cv2.putText(im, str(i), (x0 + 3, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    cv2.imwrite(out, im, [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    pdf, page, long_edge, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    cfg = eval(sys.argv[5]) if len(sys.argv) > 5 else {}
    with tempfile.TemporaryDirectory() as td:
        im = render_page(pdf, page, long_edge, td)
    res, roi, dark, im, ang = segment(im, cfg)
    print(len(res), "items", "roi", roi, "dark", dark, "angle", ang)
    draw_debug(im, res, out)
