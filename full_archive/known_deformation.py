"""Analytic nonuniform inverse-map controls; no observational validation."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'primary'))
from run_primary import grid
from flow_scaling_check import flow
OUT=None

def mapping():
    yy,xx=np.mgrid[:57,:70].astype(float)
    p=np.stack([xx,yy],axis=-1);center=np.array([34.,28.]);shift=np.array([2.,-1.])
    def rain(q):
        return (8*np.exp(-((q[...,0]-30)**2/32+(q[...,1]-24)**2/50))+
                3*np.exp(-((q[...,0]-43)**2/18+(q[...,1]-34)**2/24)))
    source=rain(p).astype('float32');rows=[]
    for name,B in [('uniform',np.zeros((2,2))),('shear',np.array([[0.,.12],[0.,0.]])),
                   ('convergence',np.array([[-.08,0.],[0.,-.08]])),
                   ('rotation_shear',np.array([[0.,-.10],[.10,0.]]))]:
        displacement=np.einsum('ij,...j->...i',B,p-center)+shift
        exact=np.einsum('ij,...j->...i',np.linalg.inv(np.eye(2)+B),p-shift+ B@center)
        approximate=p-displacement
        iterative=approximate.copy()
        for _ in range(12):iterative=p-(np.einsum('ij,...j->...i',B,iterative-center)+shift)
        assert np.max(np.abs(iterative-exact))<1e-8
        target=rain(exact)
        mask=(target>=.1)&(exact[...,0]>1)&(exact[...,0]<68)&(exact[...,1]>1)&(exact[...,1]<55)
        for method,q in [('destination_subtraction',approximate),('iterated_inverse',iterative),('exact_inverse',exact)]:
            adv=cv2.remap(source,q[...,0].astype('float32'),q[...,1].astype('float32'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            err=np.linalg.norm(q-exact,axis=-1)[mask]
            rows.append({'deformation':name,'method':method,'scored_cells':int(mask.sum()),
                         'inverse_endpoint_median_km':float(np.median(err)),
                         'inverse_endpoint_max_km':float(err.max()),
                         'residual_mae_mm':float(np.mean(np.abs(target-adv)[mask]))})
        if name=='uniform':np.testing.assert_allclose(approximate,exact,atol=1e-12)
        np.testing.assert_array_equal(
            flow.advect_forward_backward_warp(source,displacement.astype('float32')),
            cv2.remap(source,approximate[...,0].astype('float32'),approximate[...,1].astype('float32'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0))
    pd.DataFrame(rows).to_csv(OUT/'known_deformation.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();OUT=args.output;OUT.mkdir(parents=True,exist_ok=True);mapping()
