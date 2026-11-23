"""Per-strategy common mean, regime-specific variances and transitions; original B2 prediction engine."""
import numpy as np
from . import part3_b2_regime as base

def common_mean_posterior(y, states, variances, center, tau2):
    precision = 1 / tau2 + np.sum(1 / variances[states])
    variance = 1 / precision
    mean = variance * (center / tau2 + np.sum(y / variances[states]))
    return float(mean), float(variance)


def gibbs(yv, cfg, n_burn, n_keep, rng, init=None):
    center, s2 = float(yv.mean()), float(yv.var(ddof=1))
    tau2 = (cfg["sigma_mu_annual"] / 52) ** 2
    a0, b0 = cfg["var_shape"], (cfg["var_shape"] - 1) * s2
    pa, pb = cfg["p_stay_mean"] * cfg["p_stay_strength"], (1 - cfg["p_stay_mean"]) * cfg["p_stay_strength"]
    if init is None:
        mu, sig2, P = np.full(2, center), np.array([.6*s2,2*s2]), np.array([.95,.9])
    else:
        mu, sig2, P = (np.array(v,float) for v in init)
        if mu[0] != mu[1]: raise ValueError("Warm start must have common mean")
    keep=dict(mu=np.empty((n_keep,2)),s2=np.empty((n_keep,2)),P=np.empty((n_keep,2)),q1=np.empty(n_keep))
    for it in range(n_burn+n_keep):
        S,q1=base._ffbs(yv,mu,sig2,P,rng)
        mean,var=common_mean_posterior(yv,S,sig2,center,tau2)
        mu[:]=rng.normal(mean,np.sqrt(var))
        for k in (0,1):
            yk=yv[S==k]
            sig2[k]=1/rng.gamma(a0+len(yk)/2,1/(b0+((yk-mu[k])**2).sum()/2))
        n00=int(((S[:-1]==0)&(S[1:]==0)).sum());n01=int(((S[:-1]==0)&(S[1:]==1)).sum())
        n11=int(((S[:-1]==1)&(S[1:]==1)).sum());n10=int(((S[:-1]==1)&(S[1:]==0)).sum())
        P=np.array([rng.beta(pa+n00,pb+n01),rng.beta(pa+n11,pb+n10)])
        if sig2[0]>sig2[1]:mu,sig2,P,q1=mu[::-1].copy(),sig2[::-1].copy(),P[::-1].copy(),1-q1
        if it>=n_burn:
            k=it-n_burn;keep["mu"][k],keep["s2"][k],keep["P"][k],keep["q1"][k]=mu,sig2,P,q1
    return keep,(mu.copy(),sig2.copy(),P.copy())


def run_filter(*args,**kwargs):
    return base.run_filter(*args,sampler=gibbs,**kwargs)
