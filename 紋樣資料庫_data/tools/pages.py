#!/usr/bin/env python3
"""pages.py <book_id>  -- one image per chosen page (deskewed, scanner border trimmed).
Output: OUT/pages/<book>/<book>_p###.jpg  +  OUT/pitems/<book>.json"""
import sys, os, json, tempfile, time
import numpy as np, cv2
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seg

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("PATTERN_OUT", os.path.join(HERE, "out"))
SRC = os.environ.get("PATTERN_SRC", "/mnt/user-data/uploads/公眾版權圖書資料庫/公共圖書_紋樣")
FULL_LONG, Q = 2000, 82


def dark_box(im, thr=80, minfrac=0.25):
    """bounding box of the big dark photograph on an album / mount-board page (None if there is none)"""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); H, W = g.shape
    s = 800 / max(H, W); sm = cv2.resize(g, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(sm, (0, 0), 2)
    m = cv2.morphologyEx((sm < thr).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2: return None
    k = 1 + int(np.argmax(st[1:, 4])); x, y, w, h, a = st[k]
    if w * h < minfrac * sm.shape[0] * sm.shape[1]: return None
    return tuple(int(v / s) for v in (x, y, x + w, y + h))


def do_page(args):
    book, pdf, page, long_edge, cfg = args
    try:
        with tempfile.TemporaryDirectory() as td:
            im = seg.render_page(pdf, page, long_edge, td)
        im, ang = seg.deskew(im)
        H, W = im.shape[:2]
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        roi, dark = seg.find_roi(gray, 0.0)
        x0, y0, x1, y1 = roi
        keep = (x1 - x0) * (y1 - y0) >= 0.5 * W * H
        if cfg.get("roi_l") or cfg.get("roi_r"):
            x0 = max(x0, int(cfg.get("roi_l", 0) * W)); x1 = min(x1, int(cfg.get("roi_r", 1.0) * W))
        if keep:
            p = int(0.004 * max(H, W))
            x0, y0, x1, y1 = max(0, x0 - p), max(0, y0 - p), min(W, x1 + p), min(H, y1 + p)
            im = im[y0:y1, x0:x1]
        if cfg.get("crop_dark"):
            bx = dark_box(im)
            if bx:
                H2, W2 = im.shape[:2]; p = int(0.025 * max(H2, W2))
                im = im[max(0, bx[1] - p):min(H2, bx[3] + p), max(0, bx[0] - p):min(W2, bx[2] + p)]
        h, w = im.shape[:2]
        s = min(1.0, FULL_LONG / max(h, w))
        if s < 1:
            im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        d = os.path.join(OUT, "pages", book)
        os.makedirs(d, exist_ok=True)
        pid = f"{book}_p{page:03d}"
        cv2.imwrite(os.path.join(d, pid + ".jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, Q])
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        sat = float(cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 1].mean())
        return dict(id=pid, book=book, page=page, w=int(im.shape[1]), h=int(im.shape[0]), dark=bool(dark),
                    mean=round(float(g.mean()), 1), sat=round(sat, 1))
    except Exception as e:  # noqa
        return dict(id=f"{book}_p{page:03d}", book=book, page=page, err=str(e))


def main():
    book = sys.argv[1]
    b = json.load(open(os.path.join(HERE, "books.json")))[book]
    pdf = os.path.join(SRC, b["file"])
    pages = []
    for spec in b["pages"]:
        pages.extend(range(spec[0], spec[1] + 1, spec[2] if len(spec) > 2 else 1) if isinstance(spec, list) else [spec])
    tasks = [(book, pdf, p, b["long_edge"], b.get("cfg", {})) for p in pages]
    t0 = time.time(); res = []
    with Pool(int(os.environ.get("NPROC", "2"))) as pool:
        for i, r in enumerate(pool.imap_unordered(do_page, tasks)):
            res.append(r)
            if (i + 1) % 20 == 0: print(f"{book}: {i+1}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    res.sort(key=lambda r: r["id"])
    os.makedirs(os.path.join(OUT, "pitems"), exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "pitems", book + ".json"), "w"), ensure_ascii=False)
    print(book, "done", len(res), "pages; errors:", [r for r in res if "err" in r], f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
