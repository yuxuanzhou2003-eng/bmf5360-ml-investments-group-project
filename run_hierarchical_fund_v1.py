"""Fixed three-layer development trial with a self-financing daily ledger."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,brier_score_loss

TECH=['mom5','mom20','mom60','vol20','vol60','drawdown60','relative_group_mom20']
MACRO=['vix_level','vix_change_5d','dgs2_level','dgs2_change_20d','dgs10_change_20d','term_spread_10y_2y','broad_dollar_change_20d','dgs3mo_level']
def clean_json(x):
    if isinstance(x,dict):return {str(k):clean_json(v) for k,v in x.items()}
    if isinstance(x,list):return [clean_json(v) for v in x]
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x
def metrics(ret):
    ret=np.asarray(ret);nav=np.cumprod(1+ret);peak=np.maximum.accumulate(np.r_[1.,nav])[1:]
    return dict(total_return=nav[-1]-1,annualized_return=nav[-1]**(252/len(ret))-1,sharpe=float(np.mean(ret)/np.std(ret,ddof=1)*np.sqrt(252)) if np.std(ret)>0 else None,max_drawdown=float(np.min(nav/peak-1)),days=len(ret))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--panel-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args();out=a.output_dir;out.mkdir(parents=True,exist_ok=False);(out/'models').mkdir()
    stock=pd.read_csv(a.panel_dir/'stock_daily.csv',parse_dates=['Date']);mac=pd.read_csv(a.panel_dir/'macro_daily.csv',parse_dates=['Date'])
    assert not stock.duplicated(['Date','Instrument']).any() and stock.Date.le('2025-12-31').all()
    days=pd.DatetimeIndex(sorted(stock.Date.unique()));names=sorted(stock.Instrument.unique());meta=stock.drop_duplicates('Instrument').set_index('Instrument').reindex(names);groups=sorted(meta.primary_group.unique());gi=np.array([groups.index(g) for g in meta.primary_group]);D,N=len(days),len(names);G=len(groups)
    ret=stock.pivot(index='Date',columns='Instrument',values='return_decimal').reindex(index=days,columns=names)
    close=stock.pivot(index='Date',columns='Instrument',values='close').reindex(index=days,columns=names);R=ret.to_numpy();P=close.to_numpy()
    M=mac.set_index('Date').reindex(days)[MACRO].to_numpy(float)
    f=np.full((D,N,7),np.nan)
    for j in range(N):
        s=ret.iloc[:,j]
        for k,h in enumerate([5,20,60]):f[:,j,k]=(1+s).rolling(h,min_periods=h).apply(np.prod,raw=True).to_numpy()-1
        f[:,j,3]=s.rolling(20).std().to_numpy()*np.sqrt(252);f[:,j,4]=s.rolling(60).std().to_numpy()*np.sqrt(252)
        def dd(x):
            z=np.r_[1.,np.cumprod(1+x)];return np.min(z/np.maximum.accumulate(z)-1)
        f[:,j,5]=s.rolling(60).apply(dd,raw=True).to_numpy()
    basic=np.isfinite(f[:,:,:6]).all(axis=2)&np.isfinite(P)&(P>0)&np.isfinite(R)&(R>=-1)
    sf=np.full((D,G,6),np.nan)
    for i in range(D):
        for g in range(G):
            mask=basic[i]&(gi==g)
            if mask.any():
                sf[i,g]=f[i,mask,:6].mean(axis=0);f[i,mask,6]=f[i,mask,1]-sf[i,g,1]
    eligible=basic&np.isfinite(f).all(axis=2)
    industry=np.column_stack([sf.mean(axis=1),M]);industry_ok=np.isfinite(industry).all(axis=1)
    stockfeatures=pd.DataFrame(f.reshape(-1,7),columns=TECH);stockfeatures.insert(0,'Instrument',np.tile(names,D));stockfeatures.insert(0,'Date',np.repeat(days,N));stockfeatures['feature_eligible']=eligible.ravel();stockfeatures.to_csv(out/'stock_features.csv',index=False)
    sleevefeatures=pd.DataFrame(sf.reshape(-1,6),columns=TECH[:6]);sleevefeatures.insert(0,'primary_group',np.tile(groups,D));sleevefeatures.insert(0,'Date',np.repeat(days,G));sleevefeatures.to_csv(out/'sleeve_features.csv',index=False)
    pd.DataFrame(industry,index=days,columns=TECH[:6]+MACRO).to_csv(out/'industry_features.csv',index_label='Date')
    print('Features saved before labels',flush=True)
    # Future data never changes the feature universe or prediction eligibility.
    future=np.full((D,N),np.nan);slabel=np.full((D,G),np.nan);cash5=np.full(D,np.nan)
    ends=pd.Series(pd.NaT,index=days,dtype='datetime64[ns]')
    for i in range(D-6):
        block=R[i+2:i+7];ok=np.isfinite(block).all(axis=0)&(block>=-1).all(axis=0)&np.isfinite(P[i+1])&(P[i+1]>0)&np.isfinite(P[i+6])&(P[i+6]>0)
        future[i,ok]=np.prod(1+block[:,ok],axis=0)-1;ends.iloc[i]=days[i+6]
        cash5[i]=M[i,-1]/100*(days[i+6]-days[i+1]).days/365
        for g in range(G):
            mask=eligible[i]&(gi==g)
            if mask.any() and np.isfinite(future[i,mask]).all():slabel[i,g]=future[i,mask].mean()
    ailabel=slabel.mean(axis=1);iy=ailabel-cash5;sy=slabel-ailabel[:,None];xy=future-slabel[:,gi]
    train=(days>='2021-01-01')&(days<'2024-01-01')&(ends.to_numpy()<np.datetime64('2024-01-01'));dev=days>='2024-01-01';devindices=np.flatnonzero(dev)
    labels=pd.DataFrame({'Date':days,'entry_session':pd.Series(days).shift(-1),'target_end_session':ends.to_numpy(),'ai_forward_return':ailabel,'cash_forward_return':cash5,'train_date_allowed':train});labels.to_csv(out/'industry_targets.csv',index=False)
    pd.DataFrame({'Date':np.repeat(days,G),'primary_group':np.tile(groups,D),'sleeve_forward_return':slabel.ravel(),'sleeve_excess_ai':sy.ravel()}).to_csv(out/'sleeve_targets.csv',index=False)
    pd.DataFrame({'Date':np.repeat(days,N),'Instrument':np.tile(names,D),'forward_return':future.ravel(),'stock_excess_sleeve':xy.ravel()}).to_csv(out/'stock_targets.csv',index=False)
    probabilities={};modelmetrics=[];fitchecks=[]
    for family in ['logistic','random_forest']:
        probabilities[family]={}
        for layer,X,Y,ok in [('industry',industry,iy,industry_ok),('sleeve',sf,sy,np.isfinite(sf).all(axis=2)),('stock',f,xy,eligible)]:
            shape=Y.shape;xx=X.reshape(-1,X.shape[-1]);yy=Y.ravel();known=ok.ravel();tr=np.repeat(train,int(len(yy)/D))&known&np.isfinite(yy);pr=np.repeat(dev,int(len(yy)/D))&known
            model=make_pipeline(StandardScaler(),LogisticRegression(C=.5,max_iter=2000)) if family=='logistic' else RandomForestClassifier(n_estimators=200,max_depth=4,min_samples_leaf=20,max_features=.7,random_state=5360,n_jobs=-1)
            assert len(np.unique(yy[tr]>0))==2
            model.fit(xx[tr],(yy[tr]>0).astype(int));pred=np.full(len(yy),np.nan);pred[pr]=model.predict_proba(xx[pr])[:,1];probabilities[family][layer]=pred.reshape(shape)
            joblib.dump(model,out/'models'/f'{family}_{layer}.joblib')
            fitchecks.append(dict(family=family,layer=layer,train_rows=int(tr.sum()),predicted_rows=int(pr.sum()),train_last_exit=str(ends[train].max()),features=TECH if layer=='stock' else TECH[:6]+(MACRO if layer=='industry' else [])))
            ev=pr&np.isfinite(yy);y=(yy[ev]>0).astype(int)
            modelmetrics.append(dict(family=family,layer=layer,evaluated_rows=int(ev.sum()),auc=roc_auc_score(y,pred[ev]) if len(np.unique(y))==2 else None,brier=brier_score_loss(y,pred[ev])))
            keys={'Date':np.repeat(days,int(len(yy)/D)),'prediction':pred,'feature_eligible':known,'used_for_fit':tr}
            if layer!='industry':keys['entity']=np.tile(groups if layer=='sleeve' else names,D)
            pd.DataFrame(keys).to_csv(out/f'{family}_{layer}_predictions.csv',index=False)
            print(family,layer,'fit',int(tr.sum()),'predict',int(pr.sum()),flush=True)
    # Full schedule persists even for missing features; cash if gate unavailable.
    signals=devindices[::5];scheduled={int(i+1):int(i) for i in signals if i+1<D}
    variants=['equal_sleeves','momentum']+[f'{fam}_{mode}' for fam in probabilities for mode in ['timing','rotation','stock','all']]
    orders={v:{} for v in variants};orderrows=[]
    for execution,i in scheduled.items():
        available=[g for g in range(G) if eligible[i,gi==g].any()]
        for v in variants:
            w=np.zeros(N);gate=1.;gs=available[:];mode='baseline';family=None
            if v=='momentum':gs=sorted(gs,key=lambda g:(-sf[i,g,1],groups[g]))[:3]
            elif v!='equal_sleeves':
                family,mode=v.rsplit('_',1);probs=probabilities[family]
                if mode in ['timing','all']:gate=float(np.isfinite(probs['industry'][i]) and probs['industry'][i]>=.5)
                if mode in ['rotation','all']:gs=sorted(gs,key=lambda g:(-probs['sleeve'][i,g],groups[g]))[:3]
            for g in gs:
                members=np.flatnonzero(eligible[i]&(gi==g))
                if mode in ['stock','all']:
                    members=np.array(sorted(members,key=lambda j:(-probabilities[family]['stock'][i,j],names[j])))[:max(1,int(np.ceil(len(members)/2)))]
                groupbudget=min(.4,1/len(gs)) if gs else 0
                w[members]=np.minimum(.1,gate*groupbudget/len(members))
            assert w.sum()<=1+1e-12 and w.max()<=.1+1e-12
            assert all(w[gi==g].sum()<=.4+1e-12 for g in range(G))
            orders[v][execution]=w
            orderrows.extend(dict(strategy=v,formation_session=days[i],execution_session=days[execution],Instrument=names[j],target_weight=w[j],primary_group=groups[gi[j]]) for j in range(N))
    pd.DataFrame(orderrows).to_csv(out/'target_weights.csv',index=False)
    pd.DataFrame({'formation_session':days[list(scheduled.values())],'execution_session':days[list(scheduled)]}).to_csv(out/'rebalance_calendar.csv',index=False)
    print('Orders frozen; simulating cash and drifting holdings',flush=True)
    ledger=[];weights=[];gaps=[];summarymetrics=[]
    for v in variants:
        for bps in [0,10,25]:
            holdings=np.zeros(N);cash=1.;previousnav=1.;rate=bps/10000;rows=[];blocked=False
            for i in devindices:
                bad=(holdings>1e-14)&~np.isfinite(R[i])
                if bad.any():
                    gaps.append(dict(strategy=v,cost_bps=bps,Date=days[i],reason='HELD_RETURN_MISSING',instruments='|'.join(np.array(names)[bad])));blocked=True;break
                active=holdings>0;holdings[active]*=1+R[i,active]
                if not np.isfinite(M[i,-1]):gaps.append(dict(strategy=v,cost_bps=bps,Date=days[i],reason='CASH_RATE_MISSING'));blocked=True;break
                cashret=M[i,-1]/100*(days[i]-days[i-1]).days/365;cash*=1+cashret;pretrade=holdings.sum()+cash;cost=0.;turn=0.
                if i in orders[v]:
                    w=orders[v][i]
                    badtrade=(w>0)&(~np.isfinite(P[i])|(P[i]<=0))
                    if badtrade.any():gaps.append(dict(strategy=v,cost_bps=bps,Date=days[i],reason='EXECUTION_CLOSE_MISSING'));blocked=True;break
                    # Solve cost and post-cost target dollars consistently.
                    for _ in range(30):cost=rate*np.abs(w*(pretrade-cost)-holdings).sum()
                    target=w*(pretrade-cost);turn=np.abs(target-holdings).sum()/pretrade
                    assert abs(cost-rate*turn*pretrade)<1e-9
                    holdings=target;cash=(pretrade-cost)*(1-w.sum())
                nav=holdings.sum()+cash;daily=nav/previousnav-1
                record=dict(strategy=v,cost_bps=bps,Date=days[i],nav=nav,return_net=daily,turnover=turn,cost_dollars=cost,ai_weight=holdings.sum()/nav,cash_weight=cash/nav,max_stock_weight=holdings.max()/nav)
                rows.append(record);ledger.append(record)
                if bps==10:weights.extend(dict(strategy=v,Date=days[i],Instrument=names[j],weight=holdings[j]/nav) for j in range(N))
                previousnav=nav
            item=dict(strategy=v,cost_bps=bps,status='blocked' if blocked else 'complete',days=len(rows))
            if not blocked:
                item.update(metrics([r['return_net'] for r in rows]));item.update(mean_ai_weight=float(np.mean([r['ai_weight'] for r in rows])),turnover_sum=float(sum(r['turnover'] for r in rows)),max_realized_stock_weight=max(r['max_stock_weight'] for r in rows))
            summarymetrics.append(item)
    ledger=pd.DataFrame(ledger);ledger.to_csv(out/'daily_ledger.csv',index=False);pd.DataFrame(weights).to_csv(out/'daily_weights.csv',index=False);pd.DataFrame(gaps,columns=['strategy','cost_bps','Date','reason','instruments']).to_csv(out/'ledger_gaps.csv',index=False)
    pd.DataFrame(summarymetrics).to_csv(out/'performance.csv',index=False);pd.DataFrame(modelmetrics).to_csv(out/'model_metrics.csv',index=False);pd.DataFrame(fitchecks).to_csv(out/'fit_audit.csv',index=False)
    yearrows=[]
    for (v,b,y),x in ledger.groupby(['strategy','cost_bps',ledger.Date.dt.year]):yearrows.append(dict(strategy=v,cost_bps=b,year=y,**metrics(x.return_net)))
    pd.DataFrame(yearrows).to_csv(out/'yearly_performance.csv',index=False)
    boot=[];rng=np.random.default_rng(5360);wide=ledger[ledger.cost_bps.eq(10)].pivot(index='Date',columns='strategy',values='return_net')
    for v in variants:
        if v=='equal_sleeves' or len(wide)!=len(devindices) or wide[[v,'equal_sleeves']].isna().any().any():continue
        delta=(wide[v]-wide.equal_sleeves).to_numpy();L=20;n=len(delta);draw=[]
        for _ in range(500):
            starts=rng.integers(0,n-L+1,size=int(np.ceil(n/L)));idx=np.concatenate([np.arange(s,s+L) for s in starts])[:n];draw.append(delta[idx].mean())
        lo,hi=np.quantile(draw,[.025,.975]);boot.append(dict(strategy=v,mean_daily_net_increment=delta.mean(),ci_low=lo,ci_high=hi,block_length=L,repetitions=500))
    pd.DataFrame(boot).to_csv(out/'paired_block_bootstrap.csv',index=False)
    for fam in probabilities:
        wide[f'{fam}_interaction_residual']=wide[f'{fam}_all']-wide.equal_sleeves-sum(wide[f'{fam}_{mode}']-wide.equal_sleeves for mode in ['timing','rotation','stock'])
    wide.to_csv(out/'daily_ablation_attribution.csv',index_label='Date')
    checks={'no2026':days.max()<pd.Timestamp('2026-01-01'),'train_exits_before_dev':ends[train].max()<pd.Timestamp('2024-01-01'),'full_dev_calendar':len(devindices)==502,'no_ledger_gaps':not gaps,'all30_ledgers_complete':all(x['status']=='complete' for x in summarymetrics),'formation_before_execution':all(days[i]<days[e] for e,i in scheduled.items()),'weights_nonnegative':bool((pd.DataFrame(orderrows).target_weight>=0).all())}
    report=dict(status='completed_development_trial' if all(checks.values()) else 'completed_with_check_failures',checks=checks,companies=N,groups=groups,development_days=len(devindices),scheduled_orders=len(scheduled),model_metrics=modelmetrics,performance=summarymetrics,paired_bootstrap=boot,limitations=['Static current screenshot taxonomy, not historical PIT selection.','2024-25 is reused development validation, not unseen test.','US USD securities only; no claim of a global synchronized fund.','Network sleeve has one stock; no within-sleeve selection there.','Cash is a simple yield proxy; costs are assumptions, not measured impact.','Industry reference labels are uncapped equal-sleeve; implemented capped baseline can retain cash.','Daily overlapping training labels do not create IID observations.','Only fixed train then predict; no retraining or parameter search.','Stock caps apply at rebalance and drift between rebalances.'],inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.panel_dir/'stock_daily.csv',a.panel_dir/'macro_daily.csv']},script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(clean_json(report),indent=2),encoding='utf-8');print(out,flush=True);print(pd.DataFrame(summarymetrics)[lambda x:x.cost_bps.eq(10)].to_string(index=False),flush=True)
if __name__=='__main__':main()
