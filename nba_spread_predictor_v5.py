"""
NBA Spread Predictor v5 — prediction uses identical feature pipeline as training
"""
import os, json, time, warnings, requests
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams as nba_teams_static
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

SEASONS   = ["2022-23", "2023-24", "2024-25"]
ROLLING   = [5, 10, 20]
DELAY     = 0.7
ESPN_URL  = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".nba_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

STAT_COLS = ["AST","REB","FG_PCT","FG3_PCT","FT_PCT","TOV","STL","BLK","OREB","DREB"]

ESPN_TO_NBA = {
    "WSH":"WAS","NO":"NOP","NY":"NYK","GS":"GSW",
    "SA":"SAS","BKN":"BKN","CHA":"CHA","DAL":"DAL","LAC":"LAC",
}
def norm(a): return ESPN_TO_NBA.get(a.upper(), a.upper())

def _cp(k): return os.path.join(CACHE_DIR, k.replace("/","_").replace(" ","_")+".json")
def _lc(k):
    p=_cp(k)
    if os.path.exists(p) and time.time()-os.path.getmtime(p)<21600:
        with open(p) as f: return json.load(f)
def _sc(k,d):
    with open(_cp(k),"w") as f: json.dump(d,f)

def fetch_season(season):
    c=_lc(f"g_{season}")
    if c: return pd.DataFrame(c)
    print(f"  📥 {season}...")
    df=leaguegamefinder.LeagueGameFinder(
        season_nullable=season,league_id_nullable="00",
        season_type_nullable="Regular Season"
    ).get_data_frames()[0]
    time.sleep(DELAY)
    _sc(f"g_{season}", df.to_dict("records"))
    return df

# ── FEATURE ENGINEERING ──────────────────────────────────────────────────────

def _roll_feats(past_games, avail_stat):
    """Compute all rolling features from a past-games dataframe. Pure function."""
    feats = {}
    for w in ROLLING:
        window = past_games.tail(w)
        for col in avail_stat:
            v = window[col].dropna()
            feats[f"R{w}_{col}_mu"] = float(v.mean()) if len(v) else np.nan
            feats[f"R{w}_{col}_sd"] = float(v.std())  if len(v)>1 else 0.0
        if "PTS" in past_games.columns:
            pts = window["PTS"].dropna()
            feats[f"R{w}_PTS_mu"] = float(pts.mean()) if len(pts) else np.nan
            feats[f"R{w}_PTS_sd"] = float(pts.std())  if len(pts)>1 else 0.0

    if "WL" in past_games.columns and len(past_games):
        wl   = past_games["WL"].tail(10).tolist()
        last = wl[-1]
        s    = sum(1 for x in reversed(wl) if x==last)
        feats["STREAK"]       = s if last=="W" else -s
        feats["WIN_RATE_L10"] = wl.count("W")/len(wl)
        feats["WIN_RATE_L5"]  = past_games["WL"].tail(5).tolist().count("W")/min(5,len(past_games))
    else:
        feats["STREAK"]=0; feats["WIN_RATE_L10"]=0.5; feats["WIN_RATE_L5"]=0.5

    if len(past_games):
        hp = past_games[past_games["MATCHUP"].str.contains("vs\\.",na=False)]["PTS"].tail(10).dropna()
        ap = past_games[past_games["MATCHUP"].str.contains("@",   na=False)]["PTS"].tail(10).dropna()
        feats["HOME_PTS_mu"] = float(hp.mean()) if len(hp) else np.nan
        feats["AWAY_PTS_mu"] = float(ap.mean()) if len(ap) else np.nan
    else:
        feats["HOME_PTS_mu"]=np.nan; feats["AWAY_PTS_mu"]=np.nan

    return feats


def build_features(games_df):
    df = games_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["GAME_ID"]   = df["GAME_ID"].astype(str)
    df["TEAM_ID"]   = df["TEAM_ID"].astype(int)
    df = df.sort_values(["TEAM_ID","GAME_DATE","GAME_ID"]).reset_index(drop=True)
    avail = [c for c in STAT_COLS if c in df.columns]
    rows  = []

    for tid, grp in df.groupby("TEAM_ID", sort=False):
        grp = grp.sort_values("GAME_DATE").reset_index(drop=True)
        for i in range(len(grp)):
            row  = grp.iloc[i]
            past = grp.iloc[:i]
            r    = {
                "TEAM_ID":    int(tid),
                "GAME_ID":    str(row["GAME_ID"]),
                "GAME_DATE":  row["GAME_DATE"],
                "IS_HOME":    1 if "vs." in str(row.get("MATCHUP","")) else 0,
                "PTS_ACTUAL": float(row["PTS"]) if "PTS" in row.index and pd.notna(row["PTS"]) else np.nan,
                "N_PAST":     i,
                "REST":       int((row["GAME_DATE"]-grp.iloc[i-1]["GAME_DATE"]).days) if i>0 else 3,
            }
            r.update(_roll_feats(past, avail))
            rows.append(r)

    out = pd.DataFrame(rows)
    out = out[out["N_PAST"] >= 5].reset_index(drop=True)
    return out


def _add_diffs(df, prefix_h="H_", prefix_a="A_"):
    """Add H-minus-A differential columns. Works on both training and inference rows."""
    base = ([f"R{w}_{c}_mu" for w in ROLLING for c in STAT_COLS] +
            [f"R{w}_PTS_mu" for w in ROLLING] +
            ["STREAK","WIN_RATE_L10","WIN_RATE_L5","REST","IS_HOME",
             "HOME_PTS_mu","AWAY_PTS_mu"])
    for b in base:
        hc, ac = f"{prefix_h}{b}", f"{prefix_a}{b}"
        if hc in df.columns and ac in df.columns:
            df[f"D_{b}"] = df[hc] - df[ac]
    return df


def build_matchups(games_df, feat_df):
    gdf = games_df.copy()
    fdf = feat_df.copy()
    for x in [gdf, fdf]:
        x["GAME_DATE"] = pd.to_datetime(x["GAME_DATE"])
        x["GAME_ID"]   = x["GAME_ID"].astype(str)
        x["TEAM_ID"]   = x["TEAM_ID"].astype(int)

    hg = gdf[gdf["MATCHUP"].str.contains("vs\\.",na=False)].copy()
    ag = gdf[gdf["MATCHUP"].str.contains("@",   na=False)].copy()
    hf = hg.merge(fdf, on=["TEAM_ID","GAME_ID","GAME_DATE"], how="inner")
    af = ag.merge(fdf, on=["TEAM_ID","GAME_ID","GAME_DATE"], how="inner")

    skip = {"GAME_ID","GAME_DATE"}
    hf = hf.rename(columns={c:f"H_{c}" for c in hf.columns if c not in skip})
    af = af.rename(columns={c:f"A_{c}" for c in af.columns if c not in skip})

    m = hf.merge(af, on=["GAME_ID","GAME_DATE"], how="inner")
    m["MARGIN"] = m["H_PTS_ACTUAL"] - m["A_PTS_ACTUAL"]
    m = _add_diffs(m)
    m = m.dropna(subset=["MARGIN"])
    return m


def _input_cols(df):
    bad = {
        "GAME_ID","GAME_DATE","MARGIN",
        "H_PTS_ACTUAL","A_PTS_ACTUAL","H_TEAM_ID","A_TEAM_ID",
        "H_GAME_ID","A_GAME_ID","H_GAME_DATE","A_GAME_DATE",
        "H_MATCHUP","A_MATCHUP","H_SEASON","A_SEASON","H_WL","A_WL",
        "H_TEAM_NAME","A_TEAM_NAME","H_N_PAST","A_N_PAST",
        "H_PTS_ACTUAL","A_PTS_ACTUAL",
    }
    return [c for c in df.columns
            if c not in bad and "PTS_ACTUAL" not in c
            and df[c].dtype in (np.float64,np.int64,float,int)]

# ── MODEL ─────────────────────────────────────────────────────────────────────

class Model:
    def __init__(self):
        self.xgb = XGBRegressor(
            n_estimators=600, max_depth=4, learning_rate=0.02,
            subsample=0.75, colsample_bytree=0.7, min_child_weight=8,
            reg_alpha=3.0, reg_lambda=5.0, random_state=42, n_jobs=-1, verbosity=0,
        )
        self.sc   = StandardScaler()
        self.cols = None
        self.cv_mae = self.base_mae = None

    def fit(self, m):
        self.cols = _input_cols(m)
        X = m[self.cols].fillna(0).values
        y = m["MARGIN"].values
        self.base_mae = float(np.mean(np.abs(y - y.mean())))

        tscv  = TimeSeriesSplit(n_splits=5)
        fmaes = []
        for tr,va in tscv.split(X):
            sc2 = StandardScaler()
            xb  = XGBRegressor(n_estimators=600,max_depth=4,learning_rate=0.02,
                               subsample=0.75,colsample_bytree=0.7,min_child_weight=8,
                               reg_alpha=3.0,reg_lambda=5.0,random_state=42,n_jobs=-1,verbosity=0)
            xb.fit(sc2.fit_transform(X[tr]), y[tr])
            fmaes.append(mean_absolute_error(y[va], xb.predict(sc2.transform(X[va]))))

        self.cv_mae = float(np.mean(fmaes))
        self.xgb.fit(self.sc.fit_transform(X), y)
        imp = (1 - self.cv_mae/self.base_mae)*100
        print(f"  ✅ CV MAE: {self.cv_mae:.1f} pts | Baseline: {self.base_mae:.1f} pts | Improvement: {imp:.1f}%")

        sample = self.xgb.predict(self.sc.transform(X[:5]))
        print(f"  🔍 Train sample preds:   {[round(float(p),1) for p in sample]}")
        print(f"  🔍 Train sample actuals: {[round(float(v),1) for v in y[:5]]}")

    def predict_row(self, row_df):
        """row_df must have exactly the same columns as training (self.cols)."""
        X = row_df[self.cols].fillna(0).values
        p = float(self.xgb.predict(self.sc.transform(X))[0])
        c = round(min(0.9, max(0.0, 1 - self.cv_mae/self.base_mae)), 2)
        return {"margin": round(p,1), "mae": round(self.cv_mae,1), "conf": c}

# ── ESPN ──────────────────────────────────────────────────────────────────────

def fetch_espn():
    try:
        data   = requests.get(ESPN_URL, timeout=10).json()
        out    = []
        for e in data.get("events",[]):
            co   = e["competitions"][0]
            cs   = co["competitors"]
            home = next((c for c in cs if c["homeAway"]=="home"),{})
            away = next((c for c in cs if c["homeAway"]=="away"),{})
            st   = e["status"]["type"]
            spread=stm=None
            for odd in co.get("odds",[]):
                pts=odd.get("details","").split()
                if len(pts)>=2:
                    stm=pts[0]
                    for p in pts[1:]:
                        try: spread=float(p); break
                        except ValueError: pass
                    if spread is not None: break
            sit    = co.get("situation",{})
            period = sit.get("period") or co.get("status",{}).get("period") or 0
            out.append({
                "name":       e.get("shortName",""),
                "state":      st.get("state","pre"),
                "home":       norm(home.get("team",{}).get("abbreviation","")),
                "home_name":  home.get("team",{}).get("displayName",""),
                "home_score": int(home.get("score") or 0),
                "home_rec":   (home.get("records") or [{}])[0].get("summary",""),
                "away":       norm(away.get("team",{}).get("abbreviation","")),
                "away_name":  away.get("team",{}).get("displayName",""),
                "away_score": int(away.get("score") or 0),
                "away_rec":   (away.get("records") or [{}])[0].get("summary",""),
                "spread":spread,"spread_fav":stm,
                "clock":st.get("displayClock","12:00"),"period":period,
            })
        return out
    except Exception as ex:
        print(f"  ⚠️  ESPN: {ex}"); return []

# ── PREDICTION ────────────────────────────────────────────────────────────────

def _tid(abbr):
    for t in nba_teams_static.get_teams():
        if t["abbreviation"].upper()==abbr.upper(): return t["id"]

def _snap(abbr, feat_df):
    tid = _tid(abbr)
    if not tid: return None
    rows = feat_df[feat_df["TEAM_ID"]==int(tid)].sort_values("GAME_DATE")
    return rows.iloc[-1] if not rows.empty else None


def build_inference_row(home_abbr, away_abbr, feat_df):
    """
    Build a single-row DataFrame with IDENTICAL column structure to training matchups.
    This is the key fix: we construct H_*/A_*/D_* columns the same way build_matchups does.
    """
    hs = _snap(home_abbr, feat_df)
    as_ = _snap(away_abbr, feat_df)
    if hs is None or as_ is None:
        return None

    row = {}
    # Add H_ and A_ prefixed versions of every feature column
    feat_cols = [c for c in feat_df.columns
                 if c not in ("TEAM_ID","GAME_ID","GAME_DATE","PTS_ACTUAL","N_PAST")]
    for c in feat_cols:
        hv = hs[c] if c in hs.index else np.nan
        av = as_[c] if c in as_.index else np.nan
        row[f"H_{c}"] = float(hv) if pd.notna(hv) and isinstance(hv,(int,float)) else 0.0
        row[f"A_{c}"] = float(av) if pd.notna(av) and isinstance(av,(int,float)) else 0.0

    row["H_IS_HOME"] = 1.0
    row["A_IS_HOME"] = 0.0

    df_row = pd.DataFrame([row])
    df_row = _add_diffs(df_row)   # adds D_* columns exactly like build_matchups does
    return df_row


def make_pick(game, model, feat_df):
    inf_row = build_inference_row(game["home"], game["away"], feat_df)
    if inf_row is None:
        return {"error": f"Missing data: {game['home']} or {game['away']}"}

    res    = model.predict_row(inf_row)
    margin = res["margin"]
    spread = game["spread"]
    rec    = {**game, "predicted_margin":margin, "mae":res["mae"], "conf":res["conf"], "type":"pre"}

    if spread is not None:
        edge = margin - (-spread)
        rec["edge"] = round(edge,1)
        if abs(edge)<2:  rec["pick"],rec["strength"] = "PASS — edge too small","WEAK"
        elif edge>0:     rec["pick"],rec["strength"] = f"{game['home']} covers {spread:+.1f}", "STRONG" if abs(edge)>5 else "MODERATE"
        else:            rec["pick"],rec["strength"] = f"{game['away']} covers +{abs(spread):.1f}", "STRONG" if abs(edge)>5 else "MODERATE"
    else:
        rec["pick"],rec["strength"],rec["edge"] = f"No line — proj: {margin:+.1f}","N/A",None
    return rec


def make_live_pick(game, model, feat_df):
    rec = make_pick(game, model, feat_df)
    if "error" in rec: return rec
    rec["type"]="live"
    period=int(game["period"] or 1)
    try:
        pts=game["clock"].split(":"); cmins=int(pts[0])+int(pts[1])/60
    except: cmins=6.0
    played   =(period-1)*12+(12-cmins)
    remaining=max(0.0,48.0-played); pct=remaining/48.0
    live     =game["home_score"]-game["away_score"]
    blended  =round(rec["predicted_margin"]*pct+live*(1-pct),1)
    rec.update({"live_margin":live,"blended":blended,"period":period,
                "mins_left":round(remaining,1),"pct_left":round(pct*100,1)})
    spread=game["spread"]
    if spread is not None:
        le=blended-(-spread); rec["live_edge"]=round(le,1)
        rec["live_pick"]=("PASS — too close" if abs(le)<2 else
                          f"{game['home']} covers (live)" if le>0 else f"{game['away']} covers (live)")
    else: rec["live_pick"]=f"Proj: {blended:+.1f}"
    rec["alert"]=("⚠️  Big lead late" if abs(live)>15 and pct<0.25 else
                  "🔥 Nail-biter — high variance" if abs(live)<3 and pct<0.15 else "")
    return rec

# ── PRINT ─────────────────────────────────────────────────────────────────────

def pprint(r):
    print(f"\n{'─'*54}")
    print(f"🏀  {r['away_name']} ({r['away_rec']}) @ {r['home_name']} ({r['home_rec']})")
    if r["type"]=="live":
        print(f"    Q{r['period']} | {r['mins_left']:.0f}m left | {r['away']} {r['live_margin']*-1:+.0f} / {r['home']} {r['live_margin']:+.0f}")
        print(f"    Proj final (home): {r['blended']:+.1f}  |  Live edge: {r.get('live_edge','?')}")
        print(f"    📌 LIVE: {r['live_pick']}")
        if r.get("alert"): print(f"    {r['alert']}")
    else:
        print(f"    Predicted margin (home): {r['predicted_margin']:+.1f} pts  (MAE ±{r['mae']} pts)")
        if r.get("spread") is not None:
            print(f"    Vegas: {r.get('spread_fav','')} {r['spread']:+.1f}  |  Edge: {r['edge']:+.1f}")
        print(f"    📌 PICK: {r['pick']}")
        print(f"    Strength: {r['strength']}  |  Confidence: {r['conf']:.0%}")

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("\n🏀 NBA SPREAD PREDICTOR v5 — Training Pipeline")
    print("="*55)
    all_g=[]
    for s in SEASONS:
        df=fetch_season(s); df["SEASON"]=s; all_g.append(df); time.sleep(0.3)
    games_df=pd.concat(all_g,ignore_index=True)
    print(f"  📋 Records: {len(games_df):,}")

    print("\n🔧 Building features...")
    feat_df=build_features(games_df)
    print(f"  Rows: {len(feat_df):,}")

    print("\n🔗 Building matchups...")
    matchups=build_matchups(games_df,feat_df)
    m=matchups["MARGIN"]
    print(f"  Rows: {len(matchups):,} | mean={m.mean():.1f} std={m.std():.1f}")

    print("\n🤖 Training...")
    model=Model(); model.fit(matchups)

    # Verify inference pipeline matches training columns
    all_teams = feat_df["TEAM_ID"].unique()
    test_row = build_inference_row(
        next(t for t in nba_teams_static.get_teams() if t["id"]==all_teams[0])["abbreviation"],
        next(t for t in nba_teams_static.get_teams() if t["id"]==all_teams[1])["abbreviation"],
        feat_df
    )
    if test_row is not None:
        matched = sum(1 for c in model.cols if c in test_row.columns)
        print(f"  🔍 Inference col match: {matched}/{len(model.cols)} training cols found")
        if matched < len(model.cols)*0.9:
            missing = [c for c in model.cols if c not in test_row.columns]
            print(f"  ⚠️  Missing cols sample: {missing[:5]}")

    print("\n📡 ESPN games...")
    today=fetch_espn(); print(f"  Found: {len(today)}")

    recs=[]
    print("\n"+"="*55+"\n📊 PICKS\n"+"="*55)
    for g in today:
        s=g["state"]
        if s=="post": continue
        r=(make_live_pick if s=="in" else make_pick)(g,model,feat_df)
        if "error" in r: print(f"\n  ⚠️  {g['name']}: {r['error']}")
        else: recs.append(r); pprint(r)

    print(f"\n\n✅ {len(recs)} games analyzed.")
    return recs

if __name__=="__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║          NBA SPREAD PREDICTOR  v5.0                     ║
║  ⚠️  Educational purposes only — NOT financial advice   ║
╚══════════════════════════════════════════════════════════╝""")
    main()
