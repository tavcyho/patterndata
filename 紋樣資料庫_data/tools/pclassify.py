#!/usr/bin/env python3
"""pclassify.py -- category vocabulary shared by finalize.py / pack.py.
PCATS   : sidebar order of the pattern categories (a category name = pattern SHAPE; photo / drawing is a separate 'kind')
KINDS   : image types, shown at the top of the sidebar
RENAME  : old pilot-phase names -> current names"""
PCATS = ["天花板", "圓形・徽章・花飾", "線板・檐口", "柱頭・托架・建築構件", "石膏飾件", "長條・邊飾",
         "大面積・連續花紋", "角飾・單體花飾", "圖案畫・紋章・人物", "綜合圖樣頁", "活字・印刷花飾"]
KINDS = ["照片", "黑白稿", "彩色圖稿"]
RENAME = {"天花板照片": "照片", "圓形花飾照片": "圓形・徽章・花飾", "圓形・徽章黑白稿": "圓形・徽章・花飾",
          "石膏飾件照片": "石膏飾件", "柱頭・托架・建築構件照片": "柱頭・托架・建築構件",
          "長條・邊飾黑白稿": "長條・邊飾", "大面積・連續花紋黑白稿": "大面積・連續花紋",
          "角飾・單體花飾黑白稿": "角飾・單體花飾", "圖案畫・紋章・人物黑白稿": "圖案畫・紋章・人物",
          "綜合圖樣頁黑白稿": "綜合圖樣頁"}
def norm(c):
    return RENAME.get(c, c)
