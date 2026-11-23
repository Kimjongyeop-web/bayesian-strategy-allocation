"""B2.2: joint M/L regimes with regime-specific means and covariance.

Joint two-state Bayesian model with conjugate conditional sampling.
"""
import numpy as np
import pandas as pd
from numba import njit
from . import part3_b2_regime as base
from .contracts import Forecast
from .part3_b2_regime import H_DAYS
from .portfolio import month_end_mask

@njit(cache=True)
def _draw_states(log0, log1, p00, p11):
    n = len(log0)
    f = np.empty(n)
    p1 = (1-p00)/(2-p00-p11)
    for j in range(n):
        a = np.log(max(1-p1, 1e-300)) + log0[j]
        b = np.log(max(p1, 1e-300)) + log1[j]
        shift = max(a, b)
        aa, bb = np.exp(a-shift), np.exp(b-shift)
        f[j] = bb/(aa+bb)
        p1 = (1-f[j])*(1-p00)+f[j]*p11
    s = np.empty(n, dtype=np.int8)
    s[-1] = np.random.random() < f[-1]
    for j in range(n-2, -1, -1):
        if s[j+1]:
            a, b = (1-f[j])*(1-p00), f[j]*p11
        else:
            a, b = (1-f[j])*p00, f[j]*(1-p11)
        s[j] = np.random.random() < b/max(a+b, 1e-300)
    return s, f[-1]


@njit(cache=True)
def _transition(s, pa, pb):
    counts = np.zeros((2,2))
    for t in range(len(s)-1):
        counts[s[t],s[t+1]] += 1
    return np.array([np.random.beta(pa+counts[0,0],pb+counts[0,1]),
                     np.random.beta(pa+counts[1,1],pb+counts[1,0])])


@njit(cache=True)
def _inverse_wishart2(a, b, c, df):
    # Invert a Wishart draw with scale = inverse([[a,b],[b,c]]), Bartlett decomposition.
    det = a*c-b*b
    ia, ib, ic = c/det, -b/det, a/det
    l00 = np.sqrt(ia)
    l10 = ib/l00
    l11 = np.sqrt(max(ic-l10*l10, 1e-30))
    x = np.sqrt(np.random.chisquare(df))
    z = np.random.normal()
    w = np.sqrt(np.random.chisquare(df-1))
    q00 = l00*x
    q10 = l10*x+l11*z
    q11 = l11*w
    w00, w01, w11 = q00*q00, q00*q10, q10*q10+q11*q11
    dw = w00*w11-w01*w01
    return w11/dw, -w01/dw, w00/dw


@njit(cache=True)
def joint_gibbs(y, n_burn, n_keep, seed, tau2, shape, stay_mean, strength,
                old_mu, old_sig, old_p, warm):
    np.random.seed(seed)
    T = len(y)
    center = np.array([y[:,0].mean(), y[:,1].mean()])
    s2 = np.empty(2)
    for i in range(2):
        s2[i] = max(np.var(y[:,i])*T/(T-1), 1e-14)
    df0 = 2*shape+1  # p=2; marginal inverse-gamma shape = var_shape
    prior = (df0-3)*s2
    if warm:
        mu, sig, p = old_mu.copy(), old_sig.copy(), old_p.copy()
    else:
        mu = np.empty((2,2)); sig = np.zeros((2,2,2)); p = np.array([.95,.9])
        for k in range(2):
            for j in range(2):
                mu[k,j] = center[j]
                sig[k,j,j] = (.6 if k == 0 else 2.)*s2[j]
    kmu = np.empty((n_keep,2,2)); ksig = np.empty((n_keep,2,2,2))
    kp = np.empty((n_keep,2)); kq = np.empty(n_keep)
    log0, log1 = np.empty(T), np.empty(T)
    pa, pb = stay_mean*strength, (1-stay_mean)*strength
    for it in range(n_burn+n_keep):
        for k in range(2):
            a,b,c = sig[k,0,0],sig[k,0,1],sig[k,1,1]
            det = a*c-b*b
            for t in range(T):
                x,z = y[t,0]-mu[k,0],y[t,1]-mu[k,1]
                v = -.5*(np.log(det)+(c*x*x-2*b*x*z+a*z*z)/det)
                if k == 0: log0[t]=v
                else: log1[t]=v
        s,q1 = _draw_states(log0,log1,p[0],p[1])
        if it >= n_burn:
            ix=it-n_burn; kmu[ix]=mu; ksig[ix]=sig; kp[ix]=p; kq[ix]=q1
        for k in range(2):
            count=0; sx=0.; sz=0.
            for t in range(T):
                if s[t]==k:
                    count+=1; sx+=y[t,0]; sz+=y[t,1]
            a,b,c = sig[k,0,0],sig[k,0,1],sig[k,1,1]
            det=a*c-b*b
            ia,ib,ic=c/det,-b/det,a/det
            A=1/tau2+count*ia; B=count*ib; C=1/tau2+count*ic
            vdet=A*C-B*B; v00,v01,v11=C/vdet,-B/vdet,A/vdet
            bx=center[0]/tau2+ia*sx+ib*sz; bz=center[1]/tau2+ib*sx+ic*sz
            mx=v00*bx+v01*bz; mz=v01*bx+v11*bz
            l00=np.sqrt(v00); l10=v01/l00; l11=np.sqrt(max(v11-l10*l10,1e-30))
            z0,z1=np.random.normal(),np.random.normal()
            mu[k,0]=mx+l00*z0; mu[k,1]=mz+l10*z0+l11*z1
            aa,bb,cc=prior[0],0.,prior[1]
            for t in range(T):
                if s[t]==k:
                    x,z=y[t,0]-mu[k,0],y[t,1]-mu[k,1]
                    aa+=x*x; bb+=x*z; cc+=z*z
            a,b,c=_inverse_wishart2(aa,bb,cc,df0+count)
            sig[k,0,0]=a; sig[k,0,1]=b; sig[k,1,0]=b; sig[k,1,1]=c
        p = _transition(s,pa,pb)
        if np.trace(sig[0]) > np.trace(sig[1]):
            mu=mu[::-1].copy(); sig=sig[::-1].copy(); p=p[::-1].copy()
    return kmu,ksig,kp,kq,mu,sig,p




def weekly_table(y, dates, upto):
    """Existing B2 sum-of-finite-days convention, with explicit shared W-FRI keys."""
    periods=dates[:upto+1].to_period('W-FRI')
    df=pd.DataFrame(y[:upto+1],index=periods)
    return df.groupby(level=0).sum(min_count=1)

def forecast(dates, day, ids, means, logcov, V):
    a=np.asarray(means,float); B=np.asarray(logcov,float); B=(B+B.T)/2
    diag=np.diag(B)
    with np.errstate(over='ignore',invalid='ignore'):
        mean=np.exp(a+diag/2)-1
        cov=np.exp(a[:,None]+a[None,:]+(diag[:,None]+diag[None,:])/2)*np.expm1(B)
    ok=bool(np.isfinite(mean).all() and np.isfinite(cov).all() and
            np.linalg.eigvalsh(B).min()>0 and np.linalg.eigvalsh(cov).min()>0)
    return Forecast(dates[day],tuple(ids),H_DAYS,a.copy(),B.copy(),V.copy(),a,B,mean,cov,ok,
                    '' if ok else 'non-finite or non-PD predictive')

def log_moments(samples, scale, weights=None):
    """B2 scaling: mean x scale, covariance x scale; weighted version uses ddof=1 analogue."""
    x=np.asarray(samples,float)
    if x.ndim==1: x=x[:,None]
    if weights is None:
        return x.mean(axis=0)*scale, np.atleast_2d(np.cov(x,rowvar=False,ddof=1))*scale
    w=np.asarray(weights,float);w=w/w.sum()
    mean=w@x; dx=x-mean; denominator=1-w@w
    if denominator<=0: raise ValueError('posterior weights have no effective sample')
    return mean*scale, (dx.T@(dx*w[:,None]))/denominator*scale

def ess(series):
    x=np.asarray(series,float); x=x-x.mean(); n=len(x)
    v=x@x
    if n<4 or v<=0: return float(n)
    total=0.
    # Initial-positive paired autocorrelation sequence, capped at 200 lags.
    for lag in range(1,min(n-2,200),2):
        pair=(x[:-lag]@x[lag:]+x[:-(lag+1)]@x[lag+1:])/v
        if pair<=0: break
        total+=pair
    return float(min(n,n/(1+2*total)))

def simulate(keep, weeks, paths, rng):
    mu,sig,P,q=(np.repeat(keep[k],paths,axis=0) for k in ('mu','sig','P','q'))
    n=len(q); idx=np.arange(n); st=(rng.random(n)<q).astype(int)
    out=np.zeros((n,2))
    for _ in range(weeks):
        stay=np.where(st==1,P[:,1],P[:,0]);st=np.where(rng.random(n)<stay,st,1-st)
        c=sig[idx,st]; z=rng.standard_normal((n,2)); l00=np.sqrt(c[:,0,0]);l10=c[:,1,0]/l00
        l11=np.sqrt(np.maximum(c[:,1,1]-l10*l10,0))
        out[:,0]+=mu[idx,st,0]+l00*z[:,0]
        out[:,1]+=mu[idx,st,1]+l10*z[:,0]+l11*z[:,1]
    return out


def run_filter(y, dates, ids, cfg, *, prep, start, end, beta, jitter):
    if len(ids)!=2: raise ValueError('B2.2 requires exactly two strategy series')
    me=np.flatnonzero(month_end_mask(dates)); first=np.searchsorted(me,start,side='right')-1
    last=np.searchsorted(me,end,side='right')-1
    if first<0: raise ValueError('No prior month end for B2.2')
    Vs=base._ewma_V(y,prep,end,beta,jitter);scale=base.H_DAYS/(5*cfg['weeks_ahead'])
    old=(np.zeros((2,2)),np.zeros((2,2,2)),np.zeros(2)); warm=False;fc={};rows=[]
    for j,k in enumerate(range(first,last+1)):
        t=int(me[k]); table=weekly_table(y,dates,t); valid=table.dropna(how='any')
        if len(valid)<cfg['min_weeks']: raise ValueError(f'B2.2: only {len(valid)} joint weeks before {dates[t]}')
        values=valid.to_numpy(float); burn=cfg['n_burn_warm'] if warm else cfg['n_burn']
        seed=int(cfg['seed']+k)
        arr=joint_gibbs(values,burn,cfg['n_keep'],seed,(cfg['sigma_mu_annual']/52)**2,
                        cfg['var_shape'],cfg['p_stay_mean'],cfg['p_stay_strength'],*old,warm)
        mu,sig,P,q,*old=arr; warm=True
        keep={'mu':mu,'sig':sig,'P':P,'q':q};rng=np.random.default_rng(seed+7919)
        x=simulate(keep,cfg['weeks_ahead'],cfg['paths_per_draw'],rng)
        a,B=log_moments(x,scale);day=start if j==0 else t
        f=forecast(dates,day,ids,a,B,Vs[t]);fc[day]=f
        row={'date':dates[day],'origin_month_end':dates[t],'valid':f.valid,'joint_weeks':len(valid),
             'excluded_weeks':len(table)-len(valid),'last_week':str(valid.index[-1]),'p_turbulent':float(q.mean()),
             'unequal_daily_valid_count':int((np.isfinite(y[:t+1,0]) != np.isfinite(y[:t+1,1])).sum()),
             'ess_p_turbulent':ess(q),'ess_mu_M':ess(mu[:,0,0]),'ess_mu_L':ess(mu[:,0,1]),
             'ess_var_calm_M':ess(sig[:,0,0,0]),'ess_stay_calm':ess(P[:,0]),
             'rho_calm':float((sig[:,0,0,1]/np.sqrt(sig[:,0,0,0]*sig[:,0,1,1])).mean()),
             'rho_turb':float((sig[:,1,0,1]/np.sqrt(sig[:,1,0,0]*sig[:,1,1,1])).mean())}
        for i,s in enumerate(ids):
            row.update({f'mean_{s}':f.mean[i],f'var_{s}':f.cov[i,i],
                        f'mu_calm_{s}':mu[:,0,i].mean(),f'mu_turb_{s}':mu[:,1,i].mean(),
                        f'vol_calm_{s}':np.sqrt(sig[:,0,i,i]*52).mean(),
                        f'vol_turb_{s}':np.sqrt(sig[:,1,i,i]*52).mean()})
        row['cov_M_L']=f.cov[0,1];rows.append(row)
        if cfg.get('progress') and j%12==0:print('B2.2 joint',dates[t].date(),flush=True)
    return fc,pd.DataFrame(rows).set_index('date')
