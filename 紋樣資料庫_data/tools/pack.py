#!/usr/bin/env python3
"""pack.py -- build / update the offline package.
  python3 pack.py                 build every book that has pages rendered (out/pitems/<book>.json) but no record yet
  python3 pack.py --only a,b      (re)build just these books
  python3 pack.py --all           rebuild every book
  python3 pack.py --meta          only rewrite meta.js + html + manifest from the existing per-book records
Package layout (PKG = out/pkg):
  公眾版權紋樣資料庫.html
  紋樣資料庫_data/meta.js                small index, rewritten on every update
  紋樣資料庫_data/books/<book>.json      per-book record (items, categories, shard names) -> lets a later session add a book
                                         WITHOUT re-rendering the other books
  紋樣資料庫_data/thumbs/t_<book>_NNN.js base64 thumbnails   (only the new/changed book needs copying)
  紋樣資料庫_data/full/f_<book>_NNN.js   base64 full-page JPEGs
  紋樣資料庫_data/manifest.json          sha1 of every file (copy only what changed)
  紋樣資料庫_data/tools/                 the scripts + books.json/sel.json/newbooks.json (the recipe)"""
import os, sys, json, glob, base64, hashlib, shutil, datetime
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pclassify import PCATS, KINDS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("PATTERN_OUT", os.path.join(HERE, "out"))
PKG = os.path.join(OUT, "pkg")
DATA = os.path.join(PKG, "紋樣資料庫_data")
BOOKS = os.path.join(DATA, "books")
THUMB_LONG, THUMB_Q = 420, 74
TH_SHARD, FU_SHARD = 5_500_000, 9_000_000        # bytes of base64 per shard file
TOOLS = ("pages.py", "pclassify.py", "pack.py", "finalize.py", "addsel.py", "tsheet.py", "mkthumbs.py", "seg.py",
         "template.html", "books.json", "newbooks.json", "sel.json", "pcats_pilot.json", "使用說明_新增書籍.txt")
OBSOLETE = ("build.py", "classify.py", "pack_crops.py", "psheet.py")       # pilot-phase scripts, superseded


def b64(path=None, img=None):
    if img is not None:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, THUMB_Q]); raw = buf.tobytes()
    else:
        raw = open(path, "rb").read()
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode()


def write_shards(sub, book, entries, cap, fnname):
    files, cur, size, n = [], [], 0, 0

    def flush():
        nonlocal cur, size, n
        if not cur: return
        rel = f"{sub}/{fnname}_{book}_{n:03d}.js"
        os.makedirs(os.path.join(DATA, sub), exist_ok=True)
        body = "{" + ",".join(json.dumps(i) + ":" + json.dumps(u) for i, u in cur) + "}"
        with open(os.path.join(DATA, rel), "w", encoding="utf-8") as f:
            f.write(("PDBT" if sub == "thumbs" else "PDBF") + "(" + json.dumps(rel) + "," + body + ");\n")
        files.append((rel, [i for i, _ in cur])); cur, size, n = [], 0, n + 1

    for i, u in entries:
        cur.append((i, u)); size += len(u)
        if size >= cap: flush()
    flush()
    return files


def build_book(bid, order, b, cats):
    fn = os.path.join(OUT, "pitems", bid + ".json")
    its = [it for it in json.load(open(fn)) if "err" not in it and it["id"] in cats]
    if not its: return None
    for sub in ("thumbs", "full"):
        for f in glob.glob(os.path.join(DATA, sub, f"*_{bid}_[0-9][0-9][0-9].js")): os.remove(f)
    te, fe = [], []
    for it in its:
        p = os.path.join(OUT, "pages", bid, it["id"] + ".jpg")
        im = cv2.imread(p); s = THUMB_LONG / max(im.shape[:2])
        th = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))), interpolation=cv2.INTER_AREA) if s < 1 else im
        te.append((it["id"], b64(img=th))); fe.append((it["id"], b64(path=p)))
    tmap, fmap = {}, {}
    for rel, ids in write_shards("thumbs", bid, te, TH_SHARD, "t"):
        for i in ids: tmap[i] = rel
    for rel, ids in write_shards("full", bid, fe, FU_SHARD, "f"):
        for i in ids: fmap[i] = rel
    kinds = b.get("kinds", {})
    rec = dict(id=bid, order=order, title=b["title"], short=b.get("short", b["title"]), kind=b["kind"], file=b["file"],
               items=[[it["id"], it["page"], cats[it["id"]], kinds.get(str(it["page"]), b["kind"]), it["w"], it["h"], tmap[it["id"]], fmap[it["id"]]] for it in its])
    os.makedirs(BOOKS, exist_ok=True)
    json.dump(rec, open(os.path.join(BOOKS, bid + ".json"), "w"), ensure_ascii=False)
    print(f"built {bid}: {len(its)} pages", flush=True)
    return rec


def write_meta():
    recs = sorted((json.load(open(f)) for f in glob.glob(os.path.join(BOOKS, "*.json"))), key=lambda r: r["order"])
    cats = list(PCATS)
    for r in recs:
        for it in r["items"]:
            if it[2] not in cats: cats.append(it[2])
    kinds = [k for k in KINDS if any(it[3] == k for r in recs for it in r["items"])]
    kinds += sorted({it[3] for r in recs for it in r["items"]} - set(kinds))
    th, fu, thi, fui, books, items = [], [], {}, {}, [], []
    for bi, r in enumerate(recs):
        books.append(dict(id=r["id"], title=r["title"], short=r["short"], kind=r["kind"], n=len(r["items"]), file=r["file"]))
        for it in r["items"]:
            if it[6] not in thi: thi[it[6]] = len(th); th.append(it[6])
            if it[7] not in fui: fui[it[7]] = len(fu); fu.append(it[7])
            items.append([it[0], bi, it[1], cats.index(it[2]), it[4], it[5], thi[it[6]], fui[it[7]], kinds.index(it[3])])
    meta = dict(v=3, mode="page", updated=datetime.date.today().isoformat(), cats=cats, kinds=kinds, books=books, th=th, fu=fu, items=items)
    with open(os.path.join(DATA, "meta.js"), "w", encoding="utf-8") as f:
        f.write("window.PDB=" + json.dumps(meta, ensure_ascii=False, separators=(",", ":")) + ";\n")
    shutil.copy(os.path.join(HERE, "template.html"), os.path.join(PKG, "公眾版權紋樣資料庫.html"))
    if os.path.exists(os.path.join(HERE, "使用說明_新增書籍.txt")):
        shutil.copy(os.path.join(HERE, "使用說明_新增書籍.txt"), os.path.join(PKG, "使用說明_新增書籍.txt"))
    tools = os.path.join(DATA, "tools"); os.makedirs(tools, exist_ok=True)
    for f in TOOLS:
        if os.path.exists(os.path.join(HERE, f)): shutil.copy(os.path.join(HERE, f), os.path.join(tools, f))
    for f in OBSOLETE:
        open(os.path.join(tools, f), "w").write("# obsolete (pilot-phase motif cutting) -- no longer used; safe to delete\n")
    man = {}
    for root, _, fs in os.walk(PKG):
        for f in fs:
            p = os.path.join(root, f)
            man[os.path.relpath(p, PKG).replace(os.sep, "/")] = dict(sha1=hashlib.sha1(open(p, "rb").read()).hexdigest(), bytes=os.path.getsize(p))
    json.dump(man, open(os.path.join(DATA, "manifest.json"), "w"), indent=1)
    tot = sum(v["bytes"] for v in man.values())
    print(f"meta: {len(items)} pages, {len(books)} books, {len(man)} files, {tot / 1e6:.0f} MB")


def main():
    a = sys.argv[1:]
    os.makedirs(BOOKS, exist_ok=True)
    if "--meta" not in a:
        books = json.load(open(os.path.join(HERE, "books.json")))
        cats = json.load(open(os.path.join(OUT, "pcats_final.json")))
        only = set(a[a.index("--only") + 1].split(",")) if "--only" in a else None
        for order, (bid, b) in enumerate(books.items()):
            have = os.path.exists(os.path.join(BOOKS, bid + ".json"))
            if only is not None: go = bid in only
            elif "--all" in a: go = True
            else: go = not have
            if go and os.path.exists(os.path.join(OUT, "pitems", bid + ".json")):
                build_book(bid, order, b, cats)
    write_meta()


if __name__ == "__main__":
    main()
