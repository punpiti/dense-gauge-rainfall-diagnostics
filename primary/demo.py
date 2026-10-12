"""Synthetic software smoke test, not Bangkok evidence or new science."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/"analysis"))
from flow_scaling_check import scale_pair,flow_and_residual,shift_field,motion_recovery
from no_advection_ablation import operational_label

yy,xx=np.mgrid[:57,:70]
r=(.5*np.exp(-((xx-30)**2+(yy-26)**2)/32)).astype(np.float32)
n=shift_field(r,4,2)
a,b=scale_pair(r,n,"pair_max")
assert a.max()==b.max()==255
d,adv,res,mask=flow_and_residual(r,n,"pair_max")
np.testing.assert_array_equal(res,n-adv)
assert np.isfinite(d).all() and (d[~mask]==0).all()
assert operational_label(dict(peak_pos_area_cells=499,peak_pos_sum=10000,event_pos_sum=10000,n_strong_frames=20))=="other"
assert operational_label(dict(peak_pos_area_cells=500,peak_pos_sum=1000,event_pos_sum=0,n_strong_frames=0))=="high_residual"
print("Synthetic input only:",motion_recovery(r,4,2,"pair_max"))
print("PASS: common pair scaling, exact residual identity, mask, mandatory-area Boolean rule")
