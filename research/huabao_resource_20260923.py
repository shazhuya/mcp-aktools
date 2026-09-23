# -*- coding: utf-8 -*-
from __future__ import annotations
import json, math, os, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
warnings.filterwarnings("ignore")

OUT = Path("research/output_huabao_20260923")
OUT.mkdir(parents=True, exist_ok=True)
FUND_CODE="240022"
INDEX_CODE="000944"
START="20050101"
END="20260923"
INCEPTION=pd.Timestamp("2012-08-21")
ASOF=pd.Timestamp("2026-09-23")
EXPECTED_NAV=5.0070
EXPECTED_INDEX_0918=5203.40

def get_json(url, params=None, attempts=5):
    headers={"User-Agent":"Mozilla/5.0","Referer":"https://www.csindex.com.cn/"}
    last=None
    for i in range(attempts):
        try:
            r=requests.get(url,params=params,headers=headers,timeout=40)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last=e; time.sleep(2*(i+1))
    raise RuntimeError(f"GET failed {url}: {last}")

def fetch_csindex_official():
    # Official CSI endpoint, chunked by calendar year to avoid response-size/rate-limit surprises.
    url="https://www.csindex.com.cn/csindex-home/perf/index-perf"
    frames=[]
    for y in range(2005, 2027):
        s=f"{y}0101"; e=f"{y}1231"
        if y==2026: e=END
        j=get_json(url, {"indexCode":INDEX_CODE,"startDate":s,"endDate":e})
        data=j.get("data",[])
        if not data:
            raise RuntimeError(f"Official CSI returned empty data for {y}: {str(j)[:300]}")
        # CSI endpoint returns 16 fields, ordered per AKShare official adapter.
        cols=["date","index_code","cn_full","cn_short","en_full","en_short",
              "open","high","low","close","change","pct_change","volume","amount","sample_count","pe_ttm"]
        df=pd.DataFrame(data)
        if df.shape[1] != len(cols):
            raise RuntimeError(f"Unexpected CSI schema in {y}: {df.shape}")
        df.columns=cols
        df["date"]=pd.to_datetime(df["date"],errors="coerce")
        for c in ["open","high","low","close","change","pct_change","volume","amount","sample_count","pe_ttm"]:
            df[c]=pd.to_numeric(df[c],errors="coerce")
        frames.append(df)
        time.sleep(0.15)
    x=pd.concat(frames,ignore_index=True).dropna(subset=["date","close"])
    x=x.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return x

def fetch_index_eastmoney():
    # independent market-data cross-check
    import akshare as ak
    x=ak.stock_zh_index_daily_em(symbol="csi000944", start_date=START, end_date=END)
    x=x.rename(columns={"date":"date","open":"open","close":"close","high":"high","low":"low","volume":"volume","amount":"amount"})
    x["date"]=pd.to_datetime(x["date"])
    for c in ["open","close","high","low","volume","amount"]:
        if c in x.columns: x[c]=pd.to_numeric(x[c],errors="coerce")
    return x.sort_values("date").drop_duplicates("date")

def fetch_fund():
    import akshare as ak
    nav=ak.fund_open_fund_info_em(symbol=FUND_CODE, indicator="单位净值走势").copy()
    nav=nav.rename(columns={"净值日期":"date","单位净值":"unit_nav","日增长率":"daily_growth_pct"})
    nav["date"]=pd.to_datetime(nav["date"])
    nav["unit_nav"]=pd.to_numeric(nav["unit_nav"],errors="coerce")
    nav["daily_growth_pct"]=pd.to_numeric(nav["daily_growth_pct"],errors="coerce")
    nav=nav.dropna(subset=["date","unit_nav"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    acc=ak.fund_open_fund_info_em(symbol=FUND_CODE, indicator="累计净值走势").copy()
    acc=acc.rename(columns={"净值日期":"date","累计净值":"acc_nav"})
    acc["date"]=pd.to_datetime(acc["date"]); acc["acc_nav"]=pd.to_numeric(acc["acc_nav"],errors="coerce")
    acc=acc.sort_values("date").drop_duplicates("date")
    nav=nav.merge(acc[["date","acc_nav"]],on="date",how="left")
    # Daily growth published by fund data vendor is already distribution-adjusted.
    r=nav["daily_growth_pct"].fillna(0)/100.0
    nav["total_return_index"]=100*(1+r).cumprod()
    nav["cum_return_pct"]=(nav["total_return_index"]/nav["total_return_index"].iloc[0]-1)*100
    return nav

def fetch_holdings():
    import akshare as ak
    frames=[]
    for y in range(2012,2027):
        try:
            d=ak.fund_portfolio_hold_em(symbol=FUND_CODE,date=str(y)).copy()
            if d.empty: continue
            d["source_year"]=y
            frames.append(d)
        except Exception as e:
            print("HOLDINGS_WARN",y,repr(e))
        time.sleep(.12)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def clean_holdings(h):
    if h.empty: return h
    ren={"序号":"rank","股票代码":"stock_code","股票名称":"stock_name",
         "占净值比例":"weight_pct","持股数":"shares_10k","持仓市值":"value_10k","季度":"report_label"}
    h=h.rename(columns={k:v for k,v in ren.items() if k in h.columns})
    for c in ["rank","weight_pct","shares_10k","value_10k"]:
        if c in h.columns: h[c]=pd.to_numeric(h[c],errors="coerce")
    return h

def holdings_turnover(h):
    if h.empty or "report_label" not in h.columns or "stock_code" not in h.columns:
        return pd.DataFrame(), {}
    # report_label examples contain quarter/year text. Preserve source order via parsed year/quarter.
    z=h.copy()
    def parse_q(s):
        s=str(s)
        import re
        yy=re.search(r"(20\d{2})",s); q=re.search(r"([1-4])季度",s)
        return (int(yy.group(1)) if yy else 0, int(q.group(1)) if q else 0)
    z[["_y","_q"]]=pd.DataFrame(z["report_label"].map(parse_q).tolist(),index=z.index)
    rows=[]
    groups=[]
    for (y,q),g in z.groupby(["_y","_q"]):
        if y==0: continue
        g=g.sort_values("rank" if "rank" in g.columns else "weight_pct").head(10)
        codes=set(g["stock_code"].astype(str))
        weights=float(g["weight_pct"].sum()) if "weight_pct" in g.columns else np.nan
        groups.append(((y,q),codes,weights))
    groups=sorted(groups)
    prev=None
    for (yq,codes,w) in groups:
        jac=np.nan; replace=np.nan
        if prev is not None:
            inter=len(codes & prev)
            union=len(codes | prev)
            jac=inter/union if union else np.nan
            replace=1-len(codes & prev)/max(len(prev),1)
        rows.append({"year":yq[0],"quarter":yq[1],"top10_weight_pct":w,
                     "jaccard_vs_prev":jac,"replacement_rate_vs_prev":replace})
        prev=codes
    df=pd.DataFrame(rows)
    stats={}
    if not df.empty:
        stats={
          "latest_top10_weight_pct":float(df["top10_weight_pct"].dropna().iloc[-1]),
          "median_top10_replacement_rate":float(df["replacement_rate_vs_prev"].dropna().median()),
          "mean_top10_replacement_rate":float(df["replacement_rate_vs_prev"].dropna().mean()),
          "disclosure_periods":int(len(df))
        }
    return df,stats

def verify(index, index_em, fund):
    audit={}
    audit["fund_rows"]=len(fund); audit["fund_first"]=str(fund.date.min().date()); audit["fund_last"]=str(fund.date.max().date())
    audit["index_rows"]=len(index); audit["index_first"]=str(index.date.min().date()); audit["index_last"]=str(index.date.max().date())
    row=fund[fund.date==ASOF]
    audit["fund_2026_09_23_unit_nav"]=None if row.empty else float(row.unit_nav.iloc[0])
    audit["fund_nav_check_ok"]=bool(not row.empty and abs(float(row.unit_nav.iloc[0])-EXPECTED_NAV)<1e-9)
    r0918=index[index.date==pd.Timestamp("2026-09-18")]
    audit["index_2026_09_18_close"]=None if r0918.empty else float(r0918.close.iloc[0])
    audit["index_0918_check_ok"]=bool(not r0918.empty and abs(float(r0918.close.iloc[0])-EXPECTED_INDEX_0918)<0.02)
    # Cross-source overlap official CSI vs Eastmoney
    m=index[["date","close"]].merge(index_em[["date","close"]],on="date",suffixes=("_csi","_em"))
    m["diff"]=m.close_csi-m.close_em
    audit["cross_source_overlap_rows"]=len(m)
    audit["cross_source_max_abs_close_diff"]=float(m["diff"].abs().max()) if len(m) else None
    audit["cross_source_median_abs_close_diff"]=float(m["diff"].abs().median()) if len(m) else None
    # time sequence and duplicates
    audit["fund_duplicate_dates"]=int(fund.date.duplicated().sum())
    audit["index_duplicate_dates"]=int(index.date.duplicated().sum())
    # chain-return screenshot validation
    audit["fund_cum_return_pct_asof"]=float(fund.loc[fund.date==ASOF,"cum_return_pct"].iloc[0]) if not row.empty else None
    audit["fund_cum_return_vs_screenshot_439_38_diff_pp"]=None if row.empty else float(fund.loc[fund.date==ASOF,"cum_return_pct"].iloc[0]-439.38)
    return audit

def enrich_index(x):
    x=x.copy().sort_values("date").reset_index(drop=True)
    x["ret"]=x.close.pct_change()
    for w in [20,60,120,250]:
        x[f"ma{w}"]=x.close.rolling(w).mean()
    x["mom60"]=x.close/x.close.shift(60)-1
    x["mom120"]=x.close/x.close.shift(120)-1
    x["mom250"]=x.close/x.close.shift(250)-1
    x["ma120_slope60"]=x.ma120/x.ma120.shift(60)-1
    x["high250"]=x.close.rolling(250).max()
    x["dd250"]=x.close/x.high250-1
    x["vol60"]=x.ret.rolling(60).std()*np.sqrt(250)
    x["vol250"]=x.ret.rolling(250).std()*np.sqrt(250)
    # transparent state machine, intentionally strict for confirmed downtrend
    up=(x.close>x.ma250)&(x.ma60>x.ma120)&(x.ma120_slope60>0)&(x.mom120>0)
    down=(x.close<x.ma250)&(x.ma60<x.ma120)&(x.ma120_slope60<0)&(x.mom120<0)&(x.dd250<-0.12)
    bottom=(x.dd250<-0.25)&(x.mom60>x.mom120)&(x.ma120_slope60>-0.04)
    x["regime"]="震荡/过渡"
    x.loc[up,"regime"]="主升"
    x.loc[down,"regime"]="下行确认"
    x.loc[bottom & ~up & ~down,"regime"]="底部构建"
    return x

def merge_fund_index(fund,idx):
    f=fund[fund.date>=INCEPTION].copy()
    # use distribution-adjusted total return
    f["fund100"]=f.total_return_index/f.total_return_index.iloc[0]*100
    z=f.merge(idx[["date","close","ret","regime"]],on="date",how="inner")
    base=float(z.close.iloc[0]); z["resource100"]=z.close/base*100
    z["fund_ret"]=z.fund100.pct_change(); z["resource_ret"]=z.resource100.pct_change()
    for w in [60,120,250]:
        z[f"corr{w}"]=z.fund_ret.rolling(w).corr(z.resource_ret)
        z[f"beta{w}"]=z.fund_ret.rolling(w).cov(z.resource_ret)/z.resource_ret.rolling(w).var()
    z["fund_dd"]=z.fund100/z.fund100.cummax()-1
    z["resource_dd"]=z.resource100/z.resource100.cummax()-1
    return z

def monthly_analog_forecast(idx, merged, n_sims=5000, horizon_months=60, seed=20260923):
    rng=np.random.default_rng(seed)
    mi=idx.set_index("date").close.resample("ME").last().dropna().to_frame("close")
    mi["r"]=mi.close.pct_change()
    mi["mom6"]=mi.close/mi.close.shift(6)-1
    mi["mom12"]=mi.close/mi.close.shift(12)-1
    mi["high12"]=mi.close.rolling(12).max()
    mi["dd12"]=mi.close/mi.high12-1
    # nearest historical month-end analog states with at least 36m future available
    valid=mi.dropna().iloc[:-36].copy()
    cur=mi.dropna().iloc[-1]
    cols=["mom6","mom12","dd12"]
    mu=valid[cols].mean(); sd=valid[cols].std().replace(0,1)
    d=((valid[cols]-cur[cols])/sd).pow(2).sum(axis=1).pow(.5)
    analog_dates=d.nsmallest(min(18,len(d))).index
    # collect subsequent monthly returns for first 36m after analog dates
    analog_paths=[]
    for dt in analog_dates:
        loc=mi.index.get_loc(dt)
        rr=mi.r.iloc[loc+1:loc+1+36].dropna().values
        if len(rr)==36: analog_paths.append(rr)
    hist_r=mi.r.dropna().values
    if len(analog_paths)<5:
        analog_paths=[hist_r[max(0,i-36):i] for i in range(36,len(hist_r),12) if len(hist_r[max(0,i-36):i])==36]
    analog_paths=np.asarray(analog_paths)

    # fund monthly regression / historical beta distribution
    mf=merged.set_index("date")[["fund100","resource100"]].resample("ME").last().dropna()
    fr=mf.fund100.pct_change().dropna()
    rr=mf.resource100.pct_change().reindex(fr.index)
    reg=pd.concat([fr.rename("f"),rr.rename("r")],axis=1).dropna()
    # current 36m regression
    curreg=reg.tail(36)
    beta=np.cov(curreg.f,curreg.r,ddof=1)[0,1]/np.var(curreg.r,ddof=1)
    alpha=curreg.f.mean()-beta*curreg.r.mean()
    resid=(reg.f-(alpha+beta*reg.r)).dropna().values
    # beta uncertainty from rolling 24m
    betas=[]
    for i in range(24,len(reg)+1):
        g=reg.iloc[i-24:i]
        v=np.var(g.r,ddof=1)
        if v>1e-10: betas.append(np.cov(g.f,g.r,ddof=1)[0,1]/v)
    betas=np.asarray(betas) if betas else np.array([beta])

    res_paths=np.ones((n_sims,horizon_months+1))*100
    fund_paths=np.ones((n_sims,horizon_months+1))*100
    for s in range(n_sims):
        # first 36m: state-conditioned analog; last 24m: block bootstrap historical monthly returns
        a=analog_paths[rng.integers(0,len(analog_paths))] if len(analog_paths) else rng.choice(hist_r,36,replace=True)
        tail=[]
        while len(tail)<max(0,horizon_months-36):
            start=rng.integers(0,max(1,len(hist_r)-12))
            tail.extend(hist_r[start:start+12].tolist())
        path_r=np.r_[a, np.array(tail[:max(0,horizon_months-36)])][:horizon_months]
        b=float(rng.choice(betas))
        eps=rng.choice(resid,horizon_months,replace=True) if len(resid) else np.zeros(horizon_months)
        path_f=alpha+b*path_r+eps
        res_paths[s,1:]=100*np.cumprod(1+path_r)
        fund_paths[s,1:]=100*np.cumprod(1+path_f)
    qs=[10,25,50,75,90]
    out={"resource":{q:np.percentile(res_paths,q,axis=0) for q in qs},
         "fund":{q:np.percentile(fund_paths,q,axis=0) for q in qs},
         "beta_current_36m":float(beta),"alpha_monthly":float(alpha),
         "analog_dates":[str(d.date()) for d in analog_dates]}
    return out

def annualized_return(s, periods_per_year=250):
    s=pd.Series(s).dropna()
    if len(s)<2: return np.nan
    return (s.iloc[-1]/s.iloc[0])**(periods_per_year/(len(s)-1))-1

def metrics(idx, merged, hold_stats, forecast):
    latest=idx.dropna(subset=["ma250"]).iloc[-1]
    m=merged.iloc[-1]
    # period returns ending at latest common date
    def ret_days(s,dates,n):
        end=s.iloc[-1]; target=dates.iloc[-1]-pd.Timedelta(days=int(n*365.25))
        j=(dates-target).abs().idxmin()
        return end/s.loc[j]-1
    result={
      "asof_index":str(latest.date.date()),
      "resource_close":float(latest.close),
      "resource_regime":str(latest.regime),
      "resource_vs_ma250_pct":float(latest.close/latest.ma250-1),
      "resource_mom60_pct":float(latest.mom60),
      "resource_mom120_pct":float(latest.mom120),
      "resource_mom250_pct":float(latest.mom250),
      "resource_drawdown_250_pct":float(latest.dd250),
      "resource_vol60_ann":float(latest.vol60),
      "fund_latest_date":str(m.date.date()),
      "fund_unit_nav":float(m.unit_nav),
      "fund_cum_return_pct":float(m.cum_return_pct),
      "fund_drawdown_from_peak":float(m.fund_dd),
      "corr60":float(m.corr60),"corr120":float(m.corr120),"corr250":float(m.corr250),
      "beta60":float(m.beta60),"beta120":float(m.beta120),"beta250":float(m.beta250),
      "forecast_beta_current_36m":forecast["beta_current_36m"],
      "forecast_analog_dates":forecast["analog_dates"],
      "holdings":hold_stats,
    }
    return result

def plot_final(idx, fund, merged, forecast, met):
    plt.rcParams["font.sans-serif"]=["Noto Sans CJK SC","Noto Sans CJK JP","DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"]=False
    fig=plt.figure(figsize=(20,13),constrained_layout=False)
    gs=fig.add_gridspec(4,1,height_ratios=[4.5,1.7,1.7,1.35],hspace=.22)

    # Main: exact history + forecast fan. Use common rebasing at fund inception; pre-2012 resource extended consistently.
    ax=fig.add_subplot(gs[0])
    base_idx=float(idx.loc[idx.date>=INCEPTION,"close"].iloc[0])
    hist=idx[idx.date>=pd.Timestamp("2006-01-01")].copy()
    hist["resource100"]=hist.close/base_idx*100
    ax.plot(hist.date,hist.resource100,lw=1.45,label="中证内地资源 000944（每日真实收盘，2012-08-21=100）")
    ax.plot(merged.date,merged.fund100,lw=1.65,label="华宝资源优选A（每日真实总回报，2012-08-21=100）")

    # Forecast fan begins from last actual values; all forecast is dashed / shaded.
    lastd=max(idx.date.max(),fund.date.max())
    future_dates=pd.date_range(lastd+pd.offsets.MonthEnd(1),periods=60,freq="ME")
    dates=pd.DatetimeIndex([lastd]).append(future_dates)
    res0=float(hist.resource100.iloc[-1]); fund0=float(merged.fund100.iloc[-1])
    for name,fc,scale in [("资源",forecast["resource"],res0),("基金",forecast["fund"],fund0)]:
        q10=fc[10]/100*scale; q25=fc[25]/100*scale; q50=fc[50]/100*scale; q75=fc[75]/100*scale; q90=fc[90]/100*scale
        ax.fill_between(dates,q10,q90,alpha=.08)
        ax.fill_between(dates,q25,q75,alpha=.12)
        ax.plot(dates,q50,ls="--",lw=1.25,label=f"{name}统计基准情景中位数（非确定预测）")

    ax.axvline(lastd,ls=":",lw=1.2)
    ax.text(lastd,ax.get_ylim()[1]*.94,"  历史真实数据 | 右侧仅为概率预测",va="top",fontsize=11)
    # mark current
    cy=float(hist.iloc[-1].resource100)
    ax.scatter([hist.iloc[-1].date],[cy],s=85,zorder=5)
    ax.annotate(f"当前位置\n资源状态：{met['resource_regime']}\n250日高点回撤 {met['resource_drawdown_250_pct']:.1%}\n相对250日均线 {met['resource_vs_ma250_pct']:.1%}",
                xy=(hist.iloc[-1].date,cy),xytext=(pd.Timestamp("2023-01-01"),cy*1.24),
                arrowprops=dict(arrowstyle="->"),bbox=dict(boxstyle="round,pad=.45",fc="white",alpha=.9),fontsize=11)
    ax.set_title("华宝资源优选A × 中证内地资源：20年真实日频周期 + 5年概率区间",fontsize=20,fontweight="bold",pad=12)
    ax.set_ylabel("统一尺度（基金成立日=100）")
    ax.set_xlim(pd.Timestamp("2006-01-01"),pd.Timestamp("2031-10-01"))
    ax.grid(alpha=.18)
    ax.legend(ncol=2,fontsize=10,loc="upper left")

    # drawdowns
    ax2=fig.add_subplot(gs[1])
    ax2.plot(merged.date,merged.fund_dd*100,lw=1.1,label="基金从历史峰值回撤")
    ax2.plot(merged.date,merged.resource_dd*100,lw=1.1,label="资源指数从历史峰值回撤")
    ax2.axhline(-10,ls=":",lw=.8); ax2.axhline(-20,ls=":",lw=.8); ax2.axhline(-30,ls=":",lw=.8)
    ax2.set_ylabel("回撤 %"); ax2.yaxis.set_major_formatter(PercentFormatter(100))
    ax2.set_title("风险不是看‘线往下斜’，而是看真实回撤、趋势结构是否破坏",fontsize=13,fontweight="bold")
    ax2.grid(alpha=.18); ax2.legend(loc="lower left",ncol=2)

    # rolling correlation and beta
    ax3=fig.add_subplot(gs[2])
    ax3.plot(merged.date,merged.corr250,lw=1.15,label="250日滚动相关")
    ax3.plot(merged.date,merged.beta250,lw=1.15,label="250日滚动β")
    ax3.axhline(0,ls=":",lw=.8); ax3.axhline(1,ls=":",lw=.8)
    ax3.set_title("主动持仓会变化：用每日净值实际表现反推资源暴露（滚动相关 / β）",fontsize=13,fontweight="bold")
    ax3.grid(alpha=.18); ax3.legend(ncol=2)

    # decision box
    ax4=fig.add_subplot(gs[3]); ax4.axis("off")
    reg=met["resource_regime"]
    if reg=="下行确认":
        headline="当前：下行确认区 —— 风险控制优先"
        action="趋势破坏已满足严格条件；不把‘等2028’当策略，而以重新站回长期趋势作为再评估触发。"
    elif reg=="主升":
        headline="当前：主升区 —— 趋势仍占优"
        action="实际数据仍处趋势上行结构；重点监控回撤是否演化为趋势破坏，而非提前猜顶。"
    elif reg=="底部构建":
        headline="当前：底部构建区 —— 等趋势确认"
        action="低位修复已出现，但尚未等同主升；观察中长期均线与动量共振。"
    else:
        headline="当前：震荡/过渡区 —— 不是‘已确认一路跌到2028’"
        action="当前数据不支持把未来某一年当固定拐点；下一步由250日趋势、120日动量和回撤是否继续恶化决定。"
    lines=[
      headline,
      action,
      f"资源：60日动量 {met['resource_mom60_pct']:.1%}｜120日动量 {met['resource_mom120_pct']:.1%}｜相对MA250 {met['resource_vs_ma250_pct']:.1%}｜250日回撤 {met['resource_drawdown_250_pct']:.1%}",
      f"基金：当前净值 {met['fund_unit_nav']:.4f}｜累计总回报 {met['fund_cum_return_pct']:.2f}%｜当前回撤 {met['fund_drawdown_from_peak']:.1%}｜250日相关 {met['corr250']:.2f}｜β {met['beta250']:.2f}",
      "实线全部为逐交易日真实数据；虚线/阴影仅为统计情景，不是已知未来路径。"
    ]
    ax4.text(.01,.9,lines[0],fontsize=16,fontweight="bold",va="top")
    ax4.text(.01,.62,"\n".join(lines[1:]),fontsize=11.5,va="top",linespacing=1.55)
    fig.subplots_adjust(top=.95,bottom=.04,left=.07,right=.97,hspace=.30)
    fig.savefig(OUT/"华宝资源_真实日频_周期净值与决策图.png",dpi=190)
    plt.close(fig)

def main():
    print("1 fetch official CSI")
    idx=fetch_csindex_official()
    print("2 fetch eastmoney crosscheck")
    idx_em=fetch_index_eastmoney()
    print("3 fetch fund")
    fund=fetch_fund()
    print("4 fetch holdings")
    holdings=clean_holdings(fetch_holdings())
    audit=verify(idx,idx_em,fund)
    print(json.dumps(audit,ensure_ascii=False,indent=2))
    if not audit["fund_nav_check_ok"]:
        raise RuntimeError("HARD STOP: 2026-09-23 fund NAV does not match 5.0070")
    if not audit["index_0918_check_ok"]:
        raise RuntimeError("HARD STOP: official 000944 2026-09-18 close does not match 5203.40")
    idxe=enrich_index(idx)
    merged=merge_fund_index(fund,idxe)
    ht,hs=holdings_turnover(holdings)
    print("5 forecast")
    fc=monthly_analog_forecast(idxe,merged)
    met=metrics(idxe,merged,hs,fc)
    print(json.dumps(met,ensure_ascii=False,indent=2))
    print("6 save data")
    idx.to_csv(OUT/"000944_官方中证_每日真实行情.csv",index=False,encoding="utf-8-sig")
    idx_em.to_csv(OUT/"000944_东方财富交叉校验_每日行情.csv",index=False,encoding="utf-8-sig")
    fund.to_csv(OUT/"240022_每日真实净值.csv",index=False,encoding="utf-8-sig")
    merged.to_csv(OUT/"240022_vs_000944_每日合并数据.csv",index=False,encoding="utf-8-sig")
    if not holdings.empty: holdings.to_csv(OUT/"240022_历次公开持仓披露.csv",index=False,encoding="utf-8-sig")
    ht.to_csv(OUT/"240022_前十大持仓更替统计.csv",index=False,encoding="utf-8-sig")
    with open(OUT/"数据审计.json","w",encoding="utf-8") as f: json.dump(audit,f,ensure_ascii=False,indent=2)
    # serialize forecast quantiles
    fser={"beta_current_36m":fc["beta_current_36m"],"alpha_monthly":fc["alpha_monthly"],"analog_dates":fc["analog_dates"],
          "resource":{str(q):[float(v) for v in fc["resource"][q]] for q in [10,25,50,75,90]},
          "fund":{str(q):[float(v) for v in fc["fund"][q]] for q in [10,25,50,75,90]}}
    with open(OUT/"5年概率情景.json","w",encoding="utf-8") as f: json.dump(fser,f,ensure_ascii=False,indent=2)
    with open(OUT/"核心指标.json","w",encoding="utf-8") as f: json.dump(met,f,ensure_ascii=False,indent=2)
    print("7 plot")
    plot_final(idxe,fund,merged,fc,met)
    # concise markdown report
    report=f"""# 华宝资源优选A × 中证内地资源：真实日频审计报告

数据截止：{met['fund_latest_date']}

## 数据完整性
- 基金 240022：{audit['fund_rows']} 条逐交易日净值，{audit['fund_first']} 至 {audit['fund_last']}
- 中证内地资源 000944：{audit['index_rows']} 条官方逐交易日行情，{audit['index_first']} 至 {audit['index_last']}
- 2026-09-23 基金单位净值：{audit['fund_2026_09_23_unit_nav']:.4f}，与用户截图 5.0070：{'一致' if audit['fund_nav_check_ok'] else '不一致'}
- 000944 2026-09-18 收盘：{audit['index_2026_09_18_close']:.2f}，与理杏仁公开值 5203.40：{'一致' if audit['index_0918_check_ok'] else '不一致'}
- 官方中证 vs 东方财富交叉源重合 {audit['cross_source_overlap_rows']} 日，收盘差绝对值中位数 {audit['cross_source_median_abs_close_diff']:.6f}，最大值 {audit['cross_source_max_abs_close_diff']:.6f}

## 当前状态
- 资源指数状态：**{met['resource_regime']}**
- 000944 收盘：{met['resource_close']:.2f}
- 相对 250 日均线：{met['resource_vs_ma250_pct']:.2%}
- 60 日动量：{met['resource_mom60_pct']:.2%}
- 120 日动量：{met['resource_mom120_pct']:.2%}
- 250 日高点回撤：{met['resource_drawdown_250_pct']:.2%}
- 华宝资源单位净值：{met['fund_unit_nav']:.4f}
- 基金自历史峰值回撤：{met['fund_drawdown_from_peak']:.2%}
- 基金与资源指数 250 日滚动相关：{met['corr250']:.3f}
- 250 日滚动 beta：{met['beta250']:.3f}

## 持仓变化
- 披露期数量：{hs.get('disclosure_periods','NA')}
- 最近披露前十大合计：{hs.get('latest_top10_weight_pct',float('nan')):.2f}%
- 前十大季度更替率中位数：{hs.get('median_top10_replacement_rate',float('nan')):.2%}

## 预测口径
2026-09-23 之后不使用“手画周期”。未来 60 个月采用：
1. 真实 000944 月度收益历史；
2. 当前 6/12 月动量与 12 月回撤寻找历史近邻状态；
3. 状态条件化历史路径 + 12 月块自助法；
4. 基金端用真实月度回归 beta，并随机抽取历史滚动 beta 与残差，体现主动调仓和风格漂移；
5. 输出 10/25/50/75/90 分位，不给伪精确的单一路径。

历史实线是事实；未来虚线和概率带只代表统计情景。
"""
    (OUT/"研究报告.md").write_text(report,encoding="utf-8")
    print(report)

if __name__=="__main__":
    main()

# workflow trigger 2026-09-23 exact-data run
