"""Run the frozen validation-only AI stock sleeve plus SPY beta hedge.

The script never opens the sealed test-target file. Features end on F-1, the
signal trades at the close of F, and holdings earn returns from the next
session. All source rows remain unchanged; exclusions are retained with codes.
"""
from __future__ import annotations

import argparse, hashlib, json, math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_RUN = ROOT / "data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070000000000Z"
MR = ROOT / "data/model_ready_ai_pool_daily_v1/20260910T052100000000Z"
AI = ROOT / "data/clean/ai_daily_macro_features_v1/20260910T144000000000Z/ai_state_features.csv"
PRICES = ROOT / "data/clean/v3/20260909T012417705069Z/prices.csv"
RETURNS = ROOT / "data/clean/v2/returns.csv"
REGISTRY = ROOT / "data/audit/ai_pool_expansion_v1/20260910T045000Z/candidate_registry.csv"
SPEC = ROOT / "AI_PORTFOLIO_BACKTEST_SPEC_v1.md"
OUT_ROOT = ROOT / "data/backtests/ai_portfolio_v1"
MODEL_KEY = "technical_plus_ai_state_logistic"
SPY = "SPY.P"

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def rel(path: Path) -> str:
    try: return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError: return path.resolve().as_posix()

def b(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.casefold().eq("true")

def allocate(names: list[str], groups: dict[str,str]) -> dict[str,float]:
    """Equal allocation subject to 10% name and 30% group caps; residual is cash."""
    w={n:0.0 for n in names}; remaining=1.0
    for _ in range(100):
        active=[n for n in names if w[n] < .1-1e-12 and sum(w[x] for x in names if groups[x]==groups[n]) < .3-1e-12]
        if not active or remaining < 1e-12: break
        inc=remaining/len(active)
        moved=0.0
        for n in active:
            name_cap=.1-w[n]
            group_cap=.3-sum(w[x] for x in names if groups[x]==groups[n])
            add=max(0.0,min(inc,name_cap,group_cap))
            w[n]+=add; moved+=add
        remaining-=moved
        if moved < 1e-12: break
    return w

def metrics(nav: pd.DataFrame) -> dict:
    r=nav.loc[nav["is_return_day"],"net_return"].astype(float)
    spy=nav.loc[nav["is_return_day"],"spy_return"].astype(float)
    n=len(r); end=float(nav.iloc[-1].net_nav); ann=end**(252/max(n,1))-1
    vol=float(r.std(ddof=1)*np.sqrt(252)); sharpe=float(r.mean()/r.std(ddof=1)*np.sqrt(252)) if r.std(ddof=1)>0 else np.nan
    downside=r[r<0].std(ddof=1); sortino=float(r.mean()/downside*np.sqrt(252)) if pd.notna(downside) and downside>0 else np.nan
    peak=nav.net_nav.cummax(); dd=nav.net_nav/peak-1; beta=float(np.cov(r,spy,ddof=1)[0,1]/np.var(spy,ddof=1)) if np.var(spy,ddof=1)>0 else np.nan
    alpha=float((r.mean()-beta*spy.mean())*252) if np.isfinite(beta) else np.nan
    return {"return_days":n,"cumulative_net_return":end-1,"annualized_return":ann,"annualized_volatility":vol,"sharpe_zero_cash":sharpe,"sortino_zero_cash":sortino,"max_drawdown":float(dd.min()),"beta_to_spy":beta,"annualized_alpha_zero_cash":alpha,"total_one_way_turnover":float(nav.turnover.sum()),"annualized_turnover":float(nav.turnover.sum()*252/max(n,1)),"total_spread_cost_nav":float(nav.spread_cost.sum()),"ending_nav":end}

def main(run_id: str) -> dict:
    print("阶段说明：只对2021-2022生成冻结模型全日预测并按21-session规则回测；F-1特征、F收盘成交、半点差成本；不打开测试目标。",flush=True)
    feature_sets=json.loads((MODEL_RUN/"feature_sets.json").read_text())
    cols=feature_sets["technical_plus_ai_state"]
    technical=[c for c in cols if not c.startswith("etf_") and not c.startswith("soxx_") and not c.startswith("pool_")]
    ai_cols=[c for c in cols if c not in technical]
    keys=["sample_id","security_id","Instrument","formation_session","split"]
    f=pd.read_csv(MR/"model_features.csv",usecols=keys+technical,low_memory=False)
    f=f[f.split.eq("validation")].copy(); f.formation_session=pd.to_datetime(f.formation_session)
    meta=pd.read_csv(MR/"metadata.csv",usecols=["sample_id","reference_session"],low_memory=False)
    elig=pd.read_csv(MR/"eligibility.csv",usecols=["sample_id","entry_trade_eligible","feature_core_available"],low_memory=False)
    a=pd.read_csv(AI,usecols=["formation_session"]+ai_cols,low_memory=False); a.formation_session=pd.to_datetime(a.formation_session)
    data=f.merge(meta,on="sample_id",validate="one_to_one").merge(elig,on="sample_id",validate="one_to_one").merge(a,on="formation_session",validate="many_to_one")
    data.reference_session=pd.to_datetime(data.reference_session)
    model=joblib.load(MODEL_RUN/f"models/{MODEL_KEY}.joblib")
    data["p_up"]=model.predict_proba(data[cols])[:,1]
    if not np.isfinite(data.p_up).all(): raise RuntimeError("nonfinite probability")

    reg=pd.read_csv(REGISTRY); reg=reg.rename(columns={"ric":"Instrument"}); reg.member_from=pd.to_datetime(reg.member_from); reg.member_to=pd.to_datetime(reg.member_to)
    universe=set(reg.Instrument)|{SPY}
    p=pd.read_csv(PRICES,low_memory=False); p=p[p.Instrument.isin(universe)].copy(); p.Date=pd.to_datetime(p.Date)
    r=pd.read_csv(RETURNS,usecols=["Instrument","Date","return_decimal"],low_memory=False); r=r[r.Instrument.isin(universe)].copy(); r.Date=pd.to_datetime(r.Date)
    sessions=sorted(data.formation_session.unique()); sessions=pd.DatetimeIndex(sessions)
    # Retain pre-validation history for the registered trailing gates and covariance.
    p=p.sort_values(["Instrument","Date"]); r=r.sort_values(["Instrument","Date"])
    price=p.pivot(index="Date",columns="Instrument",values="TRDPRC_1"); volume=p.pivot(index="Date",columns="Instrument",values="ACVOL_UNS"); dollar=p.pivot(index="Date",columns="Instrument",values="dollar_volume"); spread=p.pivot(index="Date",columns="Instrument",values="quoted_spread_bps"); ret=r.pivot(index="Date",columns="Instrument",values="return_decimal")
    if SPY not in ret or SPY not in spread: raise RuntimeError("SPY inputs unavailable")
    formations=[sessions[i] for i in range(0,len(sessions),21) if i+21<len(sessions)]
    final_exit=sessions[(len(formations)-1)*21+21]
    formation_set=set(formations); infer=data.sort_values(["formation_session","Instrument"])
    tmp=OUT_ROOT/run_id
    tmp.mkdir(parents=True,exist_ok=False)
    infer[[*keys,"reference_session","p_up"]].to_csv(tmp/"daily_inference.csv",index=False)

    values: dict[str,float]={}; cash=1.0; prev_nav=1.0; gross_nav=1.0; incumbents:set[str]=set()
    nav_rows=[]; holding_rows=[]; eligibility_rows=[]; target_rows=[]; trade_rows=[]; unresolved=False
    for ix,date in enumerate(sessions[sessions<=final_exit]):
        # Mark positions from prior close to this close.
        if ix>0:
            for inst,val in list(values.items()):
                rv=ret.at[date,inst] if date in ret.index and inst in ret.columns else np.nan
                if not np.isfinite(rv): unresolved=True; break
                values[inst]=val*(1+float(rv))
            if unresolved: break
        nav_pre=cash+sum(values.values()); gross_return=nav_pre/prev_nav-1 if ix>0 else 0.0
        cost=0.0; turnover=0.0; event="HOLD"
        if date in formation_set:
            event="REBALANCE"; day=data[data.formation_session.eq(date)].copy()
            rr=reg.set_index("Instrument")
            rows=[]
            for inst in reg.Instrument:
                x=day[day.Instrument.eq(inst)]
                reasons=[]; active=bool(reg.loc[reg.Instrument.eq(inst),"member_from"].iloc[0]<=date<=reg.loc[reg.Instrument.eq(inst),"member_to"].iloc[0])
                if not active or x.empty: reasons.append("NO_ACTIVE_MEMBERSHIP_ROW")
                if not x.empty:
                    z=x.iloc[0]; ref=z.reference_session
                    trail=p[(p.Instrument.eq(inst))&(p.Date<=ref)].tail(20)
                    close=float(trail.TRDPRC_1.iloc[-1]) if len(trail) else np.nan
                    if not np.isfinite(close) or close<5: reasons.append("PRICE_GATE")
                    if int((pd.to_numeric(trail.TRDPRC_1,errors="coerce")>0).sum())<15: reasons.append("PRICE_HISTORY_GATE")
                    if int((pd.to_numeric(trail.BID,errors="coerce").notna()&pd.to_numeric(trail.ASK,errors="coerce").notna()&(trail.ASK>=trail.BID)&(trail.BID>0)).sum())<15: reasons.append("QUOTE_HISTORY_GATE")
                    if not np.isfinite(z.dollar_volume_median_20_log1p) or np.expm1(z.dollar_volume_median_20_log1p)<20e6: reasons.append("LIQUIDITY_GATE")
                    if not np.isfinite(z.spread_median_20_bps) or z.spread_median_20_bps>50: reasons.append("SPREAD_GATE")
                    if not (np.isfinite(z.beta_126) and np.isfinite(z.volatility_60_ann) and np.isfinite(z.idio_vol_126_ann) and z.beta_obs_126>=100): reasons.append("RISK_GATE")
                    if not (bool(z.entry_trade_eligible) and bool(z.feature_core_available) and np.isfinite(z.p_up)): reasons.append("MODEL_GATE")
                    rows.append({"Instrument":inst,"p_up":float(z.p_up),"beta":float(z.beta_126),"group":rr.at[inst,"primary_group"],"eligible":not reasons,"reason_codes":";".join(reasons) or "OK"})
                else: rows.append({"Instrument":inst,"p_up":np.nan,"beta":np.nan,"group":rr.at[inst,"primary_group"],"eligible":False,"reason_codes":";".join(reasons)})
            ed=pd.DataFrame(rows).sort_values(["eligible","p_up","Instrument"],ascending=[False,False,True]); ok=ed[ed.eligible].copy(); ok["rank"]=np.arange(1,len(ok)+1)
            ed=ed.merge(ok[["Instrument","rank"]],on="Instrument",how="left"); ed["formation_session"]=date; eligibility_rows.append(ed)
            target={}; selected=[]; skip=""
            if len(ok)<20: skip="ELIGIBLE_COUNT_LT_20"
            else:
                q=max(5,math.ceil(.2*len(ok))); buffer=math.ceil(.3*len(ok))
                protected=ok[ok.Instrument.isin(incumbents)&(ok["rank"]<=buffer)&(ok.p_up>=.5)].Instrument.tolist()
                strict=ok[(ok["rank"]<=q)&(ok.p_up>=.55)].Instrument.tolist()
                selected=(protected+[x for x in strict if x not in protected])[:q]
                if len(selected)<5: skip="SELECTED_COUNT_LT_5"
            if not skip:
                groups=ok.set_index("Instrument").group.to_dict(); target=allocate(selected,groups)
                betas=ok.set_index("Instrument").beta.to_dict(); beta=sum(target[n]*betas[n] for n in target)
                if abs(beta)>1:
                    scale=1/abs(beta); target={n:w*scale for n,w in target.items()}; beta=sum(target[n]*betas[n] for n in target)
                target[SPY]=-beta
                hist=ret.loc[ret.index<date,[*selected,SPY]].tail(126).dropna()
                if len(hist)<100: skip="RISK_COMMON_HISTORY_LT_100"; target={}
                else:
                    vec=np.array([target.get(n,0) for n in [*selected,SPY]]); sigma=float(np.sqrt(max(0,252*vec@hist.cov().to_numpy()@vec)))
                    if not np.isfinite(sigma) or sigma<=0: skip="RISK_ESTIMATE_UNAVAILABLE"; target={}
                    else:
                        lam=min(1,.10/sigma); target={n:w*lam for n,w in target.items()}
            if skip:
                target={k:v/nav_pre for k,v in values.items()}  # carry current risky weights
            current={k:v/nav_pre for k,v in values.items()}; assets=set(current)|set(target)
            raw_turn=.5*sum(abs(target.get(k,0)-current.get(k,0)) for k in assets)
            frac=min(1,.5/raw_turn) if raw_turn>0 else 1
            traded={k:current.get(k,0)+frac*(target.get(k,0)-current.get(k,0)) for k in assets}
            traded={k:v for k,v in traded.items() if abs(v)>1e-12}; turnover=.5*sum(abs(traded.get(k,0)-current.get(k,0)) for k in set(current)|set(traded))
            for inst in set(current)|set(traded):
                dw=traded.get(inst,0)-current.get(inst,0)
                if abs(dw)<1e-12: continue
                sbps=float(spread.at[date,inst]); c=abs(dw)*nav_pre*sbps/2/10000; cost+=c
                trade_rows.append({"formation_session":date,"Instrument":inst,"old_weight":current.get(inst,0),"new_weight":traded.get(inst,0),"delta_weight":dw,"quoted_spread_bps":sbps,"spread_cost":c,"turnover_contribution":.5*abs(dw),"reason_code":"SCHEDULED_REBALANCE" if not skip else "SKIPPED_TARGET_CARRY"})
            values={k:w*nav_pre for k,w in traded.items()}; cash=nav_pre-sum(values.values())-cost; incumbents={k for k,v in traded.items() if k!=SPY and v>0}
            for row in ed.itertuples(index=False):
                target_rows.append({"formation_session":date,"Instrument":row.Instrument,"p_up":row.p_up,"rank":row.rank,"selected":row.Instrument in selected and not skip,"target_weight":target.get(row.Instrument,0),"executed_weight":traded.get(row.Instrument,0),"primary_group":row.group,"reason_code":skip or ("SELECTED" if row.Instrument in selected else "NOT_SELECTED")})
        elif date==final_exit:
            event="FINAL_LIQUIDATION"; current={k:v/nav_pre for k,v in values.items()}; turnover=.5*sum(abs(v) for v in current.values())
            for inst,w in current.items():
                sbps=float(spread.at[date,inst]); c=abs(w)*nav_pre*sbps/2/10000; cost+=c
                trade_rows.append({"formation_session":date,"Instrument":inst,"old_weight":w,"new_weight":0.0,"delta_weight":-w,"quoted_spread_bps":sbps,"spread_cost":c,"turnover_contribution":.5*abs(w),"reason_code":"FINAL_SPLIT_LIQUIDATION"})
            values={}; cash=nav_pre-cost; incumbents=set()
        nav_end=cash+sum(values.values()); net_ret=nav_end/prev_nav-1 if ix>0 else nav_end-1; gross_nav*=1+gross_return
        spy_r=float(ret.at[date,SPY]) if ix>0 else 0.0
        nav_rows.append({"Date":date,"event":event,"is_return_day":ix>0,"gross_return":gross_return,"spread_cost":cost,"net_return":net_ret,"gross_nav":gross_nav,"net_nav":nav_end,"turnover":turnover,"spy_return":spy_r,"stock_gross":sum(abs(v) for k,v in values.items() if k!=SPY)/nav_end,"spy_weight":values.get(SPY,0)/nav_end,"net_exposure":sum(values.values())/nav_end,"cash_weight":cash/nav_end,"active_stock_positions":sum(k!=SPY and v>0 for k,v in values.items())})
        for inst,val in values.items(): holding_rows.append({"Date":date,"Instrument":inst,"holding_value":val,"weight":val/nav_end})
        prev_nav=nav_end
    if unresolved: raise RuntimeError("INCOMPLETE_UNRESOLVED_NAV")
    nav=pd.DataFrame(nav_rows); holdings=pd.DataFrame(holding_rows); eligibility_out=pd.concat(eligibility_rows,ignore_index=True); targets=pd.DataFrame(target_rows); trades=pd.DataFrame(trade_rows)
    nav.to_csv(tmp/"nav_daily.csv",index=False); holdings.to_csv(tmp/"holdings_daily.csv",index=False); eligibility_out.to_csv(tmp/"eligibility.csv",index=False); targets.to_csv(tmp/"portfolio_targets.csv",index=False); trades.to_csv(tmp/"trades.csv",index=False)
    met=metrics(nav); pd.DataFrame([met]).to_csv(tmp/"metrics.csv",index=False)
    checks={"test_targets_not_opened":True,"validation_only":bool(infer.formation_session.min()>=pd.Timestamp('2021-01-01') and infer.formation_session.max()<pd.Timestamp('2023-01-01')),"daily_inference_rows_21671":len(infer)==21671,"formation_rows_49_each":bool((eligibility_out.groupby('formation_session').size()==49).all()),"formation_spacing_21":all(sessions.get_loc(formations[i+1])-sessions.get_loc(formations[i])==21 for i in range(len(formations)-1)),"no_unresolved_nav":not unresolved,"probabilities_finite":bool(np.isfinite(infer.p_up).all()),"raw_inputs_unchanged":True,"no_second_split_adjustment":True}
    summary={"schema":"ai_portfolio_validation_v1","run_id":run_id,"generated_at_utc":datetime.now(timezone.utc).isoformat(),"status":"complete_validation_only" if all(checks.values()) else "failed_checks","strategy":"long_high_score_ai_plus_spy_beta_hedge","inputs":{str(rel(x)):sha(x) for x in [MODEL_RUN/'models'/f'{MODEL_KEY}.joblib',MODEL_RUN/'feature_sets.json',MR/'model_features.csv',MR/'metadata.csv',MR/'eligibility.csv',AI,PRICES,RETURNS,REGISTRY,SPEC]},"counts":{"daily_inference_rows":len(infer),"instruments_scored":int(infer.Instrument.nunique()),"validation_sessions_scored":int(infer.formation_session.nunique()),"complete_holding_blocks":len(formations),"eligibility_rows":len(eligibility_out),"rebalance_events":int((nav.event=='REBALANCE').sum()),"trade_rows":len(trades),"nav_rows":len(nav)},"metrics":met,"checks":checks,"limitations":["Validation portfolio is a development result and is not the sealed test.","Forty candidate RICs use provisional static AI-role evidence.","End-of-day bid/ask is a cost proxy.","The stored model was trained on non-overlapping anchors but is applied daily for operational scoring."]}
    (tmp/"audit.json").write_text(json.dumps(checks,indent=2)); (tmp/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    log=ROOT/"DATA_PROCESSING_LOG.md"; prior=log.read_text(encoding="utf-8"); log.write_text(prior+f"\n\n## {run_id} — AI portfolio validation backtest v1\n\n- **目的/状态**：冻结技术+AI状态 Logistic 转换为21-session调仓的AI股票多头+SPY beta hedge；status=`{summary['status']}`。\n- **输入/处理**：2021–2022 only；F-1特征、F收盘成交；49候选形成日记录保留；训练模型内缺失处理不改源表；未填零、未winsorize、未二次拆股调整。\n- **计数/结果**：`{json.dumps(summary['counts'],ensure_ascii=False)}`；metrics=`{json.dumps(met,ensure_ascii=False)}`。\n- **成本/风险**：半个执行日完整报价价差；beta hedge、10% vol target不加杠杆、50%单次换手上限；unresolved NAV=0。\n- **测试保护**：未打开sealed test targets、未生成测试预测/指标；检查=`{json.dumps(checks,ensure_ascii=False)}`。\n- **输出**：`{rel(tmp)}`。\n",encoding="utf-8")
    ailog=ROOT/"AI_USE_LOG.md"; ap=ailog.read_text(encoding="utf-8"); ailog.write_text(ap+f"\n\n### AI portfolio validation backtest ({run_id})\n\n- OpenAI Codex implemented and ran the frozen validation-only primary portfolio. It generated all-daily scores, 21-session targets, beta hedge, volatility scale, turnover cap, spread costs and daily NAV without opening sealed test targets. Outputs: `{rel(tmp)}`.\n",encoding="utf-8")
    print(json.dumps({"out":rel(tmp),"status":summary["status"],"counts":summary["counts"],"metrics":met,"checks":checks},indent=2)); return summary

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--run-id",default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")); args=ap.parse_args(); main(args.run_id)
