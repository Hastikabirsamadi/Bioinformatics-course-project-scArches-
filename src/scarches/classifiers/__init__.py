# scHPL is only needed for the treeArches classifier, not for expiMap. Newer scikit-learn versions
# make its import fail, so a failed import must not stop the rest of scarches from loading.
try:
    import scHPL
except Exception:
    scHPL = None
