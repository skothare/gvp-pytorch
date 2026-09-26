import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'benchmarks/atom3d_lba'))
ESTATS=Path(os.environ.get('LBA_ESTATS_ROOT',str(ROOT.parent/'Estats')))
sys.path.insert(0,str(ESTATS/'benchmarks/atom3d_lba'))
