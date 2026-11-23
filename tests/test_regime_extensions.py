import copy
import numpy as np
import pandas as pd
from sprint2.part3_b2_1_shared_mean import common_mean_posterior
from sprint2.part3_b2_2_joint import joint_gibbs
from sprint2.run import pipeline,load_config


def test_selected_policy_set():
    assert [p['id'] for p in load_config()['policies']]==[
        'M_only','L_only','EW_monthly','Cash_timing','B0','B1','B2','B2_1','B2_2']


def test_shared_mean_conditional():
    y=np.array([1.,2.,3.]);state=np.array([0,1,1]);var=np.array([1.,4.])
    mean,variance=common_mean_posterior(y,state,var,0.,2.)
    np.testing.assert_allclose([mean,variance],[(1+2/4+3/4)/2,1/2])


def test_joint_sampler_pd_and_identified():
    y=np.random.default_rng(23).normal(0,.02,(120,2))
    mu,sig,p,q,*_=joint_gibbs(y,70,120,42,(.2/52)**2,3.,.96,20.,np.zeros((2,2)),np.zeros((2,2,2)),np.zeros(2),False)
    assert np.linalg.eigvalsh(sig).min()>0
    assert (np.trace(sig[:,0],axis1=1,axis2=2)<=np.trace(sig[:,1],axis1=1,axis2=2)).all()
    assert (mu[:,0,0]!=mu[:,0,1]).any() and ((q>=0)&(q<=1)).all()


def test_original_seven_unchanged(market,cfg,result):
    c=copy.deepcopy(cfg);c['policies']=[p for p in c['policies'] if p['id'] not in ['B2_1','B2_2']]
    reference=pipeline(c,market)
    for name,old in reference['policies'].items():
        pd.testing.assert_frame_equal(old['daily'],result['policies'][name]['daily'])


def test_selected_forecasts_causal(market,cfg):
    from test_sprint2 import perturbed
    c=copy.deepcopy(cfg);c['policies']=[p for p in c['policies'] if p['id'] in ['B2_1','B2_2']]
    a=pipeline(c,market);b=pipeline(c,perturbed(market,350));cut=market.dates[350]
    for name in ['B2_1','B2_2']:
        pd.testing.assert_frame_equal(a['forecasts'][name].loc[:cut],b['forecasts'][name].loc[:cut])
        pd.testing.assert_frame_equal(a['policies'][name]['decisions'].loc[:cut],b['policies'][name]['decisions'].loc[:cut])
