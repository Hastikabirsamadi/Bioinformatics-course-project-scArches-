"""Download the tutorial data (same Google-Drive ids as the scArches expiMap notebooks) into data/."""
import os
import gdown

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "pbmc_tutorial.h5ad": "1Rnm-XKEqPLdOq3lpa3ka2aV4bOXVCLP0",
    "kang_tutorial.h5ad": "1t3oMuUfueUz_caLm5jmaEYjBxVNSsfxG",
    "reactome.gmt": "1136LntaVr92G1MphGeMVcmpE0AqcqM6c",
}
os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
for name, fid in FILES.items():
    path = os.path.join(HERE, "data", name)
    if os.path.exists(path):
        print("have", name)
    else:
        gdown.download(f"https://drive.google.com/uc?id={fid}", path, quiet=False)
