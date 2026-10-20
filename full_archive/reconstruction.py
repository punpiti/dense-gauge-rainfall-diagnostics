"""Same Gaussian arithmetic, with station kernels shared across timestamps.

The station summation order and float64 accumulators are retained. This is a
computational optimization, not a different reconstruction method. Validate
against the original frame-by-frame implementation before scientific use.
"""
from functools import lru_cache
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"primary"))
from run_primary import grid
STATIONS=None
BOUNDARY=None

def configure(stations,boundary):
    global STATIONS,BOUNDARY
    STATIONS=Path(stations);BOUNDARY=Path(boundary)
    geometry.cache_clear();kernels.cache_clear()


@lru_cache(maxsize=1)
def geometry():
    if STATIONS is None or BOUNDARY is None:raise RuntimeError('Configure explicit reconstruction coordinates and boundary first')
    st=grid.load_station_locations(str(STATIONS))
    st=grid.project_stations_to_utm(st)
    bounds,_=grid.load_boundary_bounds_wgs84(str(BOUNDARY))
    xe,ye=grid.build_grid_edges_from_bounds(bounds,1000.,2000.)
    return st,xe,ye


@lru_cache(maxsize=32)
def kernels(sigma,cutoff_km=None):
    st,xe,ye=geometry();xc=(xe[:-1]+xe[1:])/2;yc=(ye[:-1]+ye[1:])/2
    radius=(3*float(sigma) if cutoff_km is None else float(cutoff_km))*1000
    cutoff2=radius*radius;two_sigma2=2*(float(sigma)*1000)**2
    result={}
    for row in st.itertuples(index=False):
        x0,y0=row.x,row.y
        xlo,xhi=np.searchsorted(xc,[x0-radius,x0+radius],side='left')
        # Original upper bound includes a center exactly at the cutoff.
        xhi=np.searchsorted(xc,x0+radius,side='right')
        ylo=np.searchsorted(yc,y0-radius,side='left');yhi=np.searchsorted(yc,y0+radius,side='right')
        dx=xc[xlo:xhi][None,:]-x0;dy=yc[ylo:yhi][:,None]-y0;d2=dx*dx+dy*dy
        weights=np.exp(-d2/two_sigma2,dtype=np.float64);weights[d2>cutoff2]=0
        result[int(row.stationid)]=(int(ylo),int(yhi),int(xlo),int(xhi),weights)
    return result


def reconstruct(cache,sigma=2.5,anchor=True,cutoff_km=None):
    df=cache.copy();df['time']=pd.to_datetime(df.time)
    df['stationid']=grid.normalize_stationid_to_int(df.stationid)
    df=df.dropna(subset=['time','stationid']);df['rainfall']=pd.to_numeric(df.rainfall,errors='coerce')
    station_order=df.stationid.drop_duplicates().astype(int).tolist()
    times=sorted(df.time.unique().tolist());st,xe,ye=geometry()
    nt,ny,nx=len(times),len(ye)-1,len(xe)-1
    observations=df.pivot(index='time',columns='stationid',values='rainfall').reindex(index=times)
    num=np.zeros((nt,ny,nx),dtype=np.float64);den=np.zeros_like(num)
    kernel=kernels(float(sigma),cutoff_km)
    for station in station_order:
        if station not in kernel:continue
        y0,y1,x0,x1,w=kernel[station]
        values=observations[station].to_numpy(dtype=float);finite=np.isfinite(values)
        num[:,y0:y1,x0:x1]+=w[None,:,:]*np.where(finite,values,0.)[:,None,None]
        den[:,y0:y1,x0:x1]+=w[None,:,:]*finite[:,None,None]
    out=np.full(num.shape,np.nan,dtype=np.float32);valid=den>0
    out[valid]=(num[valid]/den[valid]).astype(np.float32)
    if anchor:
        merged=df.merge(st[['stationid','x','y']],on='stationid',how='inner')
        merged['ix']=np.searchsorted(xe,merged.x.to_numpy(),side='right')-1
        merged['iy']=np.searchsorted(ye,merged.y.to_numpy(),side='right')-1
        merged=merged.loc[merged.ix.between(0,nx-1)&merged.iy.between(0,ny-1)]
        means=merged.groupby(['time','iy','ix']).rainfall.mean().dropna().reset_index()
        ti=pd.Index(times).get_indexer(means.time)
        out[ti,means.iy.to_numpy(),means.ix.to_numpy()]=means.rainfall.to_numpy(dtype=np.float32)
    for i in range(nt):out[i]=grid.apply_min_rain(out[i],.1,False)
    return out,np.array(times,dtype='datetime64[ns]'),(xe[:-1]+xe[1:])/2,(ye[:-1]+ye[1:])/2
