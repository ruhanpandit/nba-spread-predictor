"""
NBA Spread Predictor v3 — clean rebuild, no data leakage
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

# ── CONFIG ───────────────────────────────────────────────────────────────────
SEASONS      = ["2022-23", "2023-24", "2024-25"]
ROLLING      = [5, 10, 20]
DELAY        = 0.7
ESPN_URL     = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
CACHE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".nba_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Stats that are known BEFORE the game ends (no outcome leakage)
# PTS is intentionally excluded here — it IS the thing we're predicting
STAT_COLS = ["AST", "REB", "FG_PCT", "FG3_PCT", "FT_PCT", "TOV", "STL", "BLK", "OREB", "DREB"]

# ESPN abbreviation → nba_api abbreviation fixes
ESPN_TO_NBA = {
    "WSH": "WAS", "NO":  "NOP", "NY":  "NYK",
    "GS":  "GSW", "SA":  "SAS", "LAC": "LAC",
    "DAL": "DAL", "BKN": "BKN", "CHA": "CHA",
}

def norm_abbr(abbr: str) -> str:
    return ESPN_TO_NBA.get(abbr.upper(), abbr.upper())

# ── CACHE ────────────────────────────────────────────────────────────────────
def _cache_path(key):
    return os.path.join(CACHE_DIR, key.replace("/","_").replace(" ","_") + ".json")

def _load_cache(key):
    p = _cache_path(key)
    if os.path.exists(p) and time.time() - os.path.getmtime(p) < 21600:
        with open(p) as f: return json.load(f)
    return None

def _save_cache(key, data):
    with open(_cache_path(key), "w") as f: json.dump(data, f)

# ── DATA FETCH ───────────────────────────────────────────────────────────────
def fetch_season_games(season):
    cached = _load_cache(f"games_{season}")
    if cached: return pd.DataFrame(cached)
    print(f"  📥 Fetching {season}...")
    df = leaguegamefinder.LeagueGameFinder(
        season_nullable=season, league_id_nullable="00",
        season_type_nullable="Regular Season"
    ).get_data_frames()[0]
    time.sleep(DELAY)
    _save_cache(f"games_{season}", df.to_dict("records"))
    return df

# ── FEATURE ENGINEERING ──────────────────────────────────────────────────────
def build_features(games_df):
    """
    Build one row per game per team with ONLY pre-game information.
    Key rules:
      - past = games BEFORE current game (iloc[:i])
      - PTS_ACTUAL stored separately, never used as input feature
      - PLUS_MINUS excluded (it equals PTS difference, pure leakage)
    """
    df = games_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    rows = []
    for tid, grp in df.groupby("TEAM_ID"):
        grp = grp.sort_values("GAME_DATE").reset_index(drop=True)
        avail = [c for c in STAT_COLS if c in grp.columns]

        for i in range(len(grp)):
            row  = grp.iloc[i]
            past = grp.iloc[:i]   # STRICTLY past games only

            feat = {
                "TEAM_ID":    tid,
                "GAME_ID":    row["GAME_ID"],
                "GAME_DATE":  row["GAME_DATE"],
                "IS_HOME":    1 if "vs." in str(row.get("MATCHUP","")) else 0,
                # PTS_ACTUAL is the label source — stored but NOT used as input
                "PTS_ACTUAL": float(row["PTS"]) if "PTS" in row and pd.notna(row["PTS"]) else np.nan,
                "N_PAST":     i,
            }

            for w in ROLLING:
                window = past.tail(w)
                for col in avail:
                    v = window[col].dropna()
                    feat[f"R{w}_{col}_avg"] = float(v.mean()) if len(v) else np.nan
                    feat[f"R{w}_{col}_std"] = float(v.std())  if len(v) > 1 else 0.0

                # Rolling PTS separately (needed for offense rating)
                if "PTS" in grp.columns:
                    pts = window["PTS"].dropna()
                    feat[f"R{w}_PTS_avg"] = float(pts.mean()) if len(pts) else np.nan
                    feat[f"R{w}_PTS_std"] = float(pts.std())  if len(pts) > 1 else 0.0

            # Win rate & streak (from past only)
            if "WL" in grp.columns and len(past):
                wl = past["WL"].tail(10).tolist()
                feat["WIN_RATE_L10"] = wl.count("W") / len(wl)
                streak, last = 0, wl[-1]
                for r in reversed(wl):
                    if r == last: streak += 1
                    else: break
                feat["STREAK"] = streak if last == "W" else -streak
            else:
                feat["WIN_RATE_L10"] = 0.5
                feat["STREAK"]       = 0

            # Home/away performance split
            if len(past):
                h_pts = past[past["MATCHUP"].str.contains("vs\\.", na=False)]["PTS"].tail(10)
                a_pts = past[past["MATCHUP"].str.contains("@",    na=False)]["PTS"].tail(10)
                feat["HOME_PTS_avg"] = float(h_pts.mean()) if len(h_pts) else np.nan
                feat["AWAY_PTS_avg"] = float(a_pts.mean()) if len(a_pts) else np.nan
            else:
                feat["HOME_PTS_avg"] = np.nan
                feat["AWAY_PTS_avg"] = np.nan

            feat["REST_DAYS"] = int((row["GAME_DATE"] - grp.iloc[i-1]["GAME_DATE"]).days) if i > 0 else 3

            rows.append(feat)

    out = pd.DataFrame(rows)
    # Drop rows where we have almost no past data (first 4 games of season)
    out = out[out["N_PAST"] >= 4].reset_index(drop=True)
    return out

def build_matchups(games_df, feat_df):
    """Merge home & away feature rows into one matchup row. Target = home margin."""
    games_df  = games_df.copy()
    feat_df   = feat_df.copy()
    games_df["GAME_DATE"]  = pd.to_datetime(games_df["GAME_DATE"])
    feat_df["GAME_DATE"]   = pd.to_datetime(feat_df["GAME_DATE"])

    home_g = games_df[games_df["MATCHUP"].str.contains("vs\\.", na=False)].copy()
    away_g = games_df[games_df["MATCHUP"].str.contains("@",    na=False)].copy()

    hf = home_g.merge(feat_df, on=["TEAM_ID","GAME_ID","GAME_DATE"], how="inner")
    af = away_g.merge(feat_df, on=["TEAM_ID","GAME_ID","GAME_DATE"], how="inner")

    skip = {"GAME_ID","GAME_DATE"}
    hf = hf.rename(columns={c: f"H_{c}" for c in hf.columns if c not in skip})
    af = af.rename(columns={c: f"A_{c}" for c in af.columns if c not in skip})

    m = hf.merge(af, on=["GAME_ID","GAME_DATE"], how="inner")

    # Target: actual home margin
    m["MARGIN"] = m["H_PTS_ACTUAL"] - m["A_PTS_ACTUAL"]

    # Differential features — most important signal
    feat_cols = [c.replace("H_","") for c in hf.columns
                 if c.startswith("H_R") or c.startswith("H_WIN") or c.startswith("H_STREAK")]
    for col in feat_cols:
        hc, ac = f"H_{col}", f"A_{col}"
        if hc in m.columns and ac in m.columns:
            m[f"D_{col}"] = m[hc] - m[ac]

    if "H_REST_DAYS" in m.columns and "A_REST_DAYS" in m.columns:
        m["D_REST"] = m["H_REST_DAYS"] - m["A_REST_DAYS"]

    m = m.dropna(subset=["MARGIN"])
    return m

def get_input_cols(df):
    """Input features: everything except identifiers and the target."""
    bad = {
        "GAME_ID","GAME_DATE","MARGIN",
        "H_PTS_ACTUAL","A_PTS_ACTUAL",
        "H_TEAM_ID","A_TEAM_ID",
        "H_GAME_ID","A_GAME_ID",
        "H_GAME_DATE","A_GAME_DATE",
        "H_MATCHUP","A_MATCHUP",
        "H_SEASON","A_SEASON",
        "H_WL","A_WL","H_TEAM_NAME","A_TEAM_NAME",
    }
    return [c for c in df.columns
            if c not in bad
            and df[c].dtype in (np.float64, np.int64, float, int)
            and "PTS_ACTUAL" not in c]

# ── MODEL ────────────────────────────────────────────────────────────────────
class SpreadPredictor:
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=600, max_depth=4, learning_rate=0.02,
            subsample=0.75, colsample_bytree=0.7,
            min_child_weight=8, reg_alpha=3.0, reg_lambda=5.0,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        self.scaler = StandardScaler()
        self.cols   = None
        self.trained = False
        self.cv_mae  = None
        self.base_mae = None

    def fit(self, matchups):
        self.cols = get_input_cols(matchups)
        X = matchups[self.cols].fillna(0).values
        y = matchups["MARGIN"].values
        self.base_mae = float(np.mean(np.abs(y - y.mean())))

        # Time-series CV
        tscv = TimeSeriesSplit(n_splits=5)
        fold_maes = []
        for tr, va in tscv.split(X):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr]); Xva = sc.transform(X[va])
            m = XGBRegressor(
                n_estimators=600, max_depth=4, learning_rate=0.02,
                subsample=0.75, colsample_bytree=0.7,
                min_child_weight=8, reg_alpha=3.0, reg_lambda=5.0,
                random_state=42, n_jobs=-1, verbosity=0,
            )
            m.fit(Xtr, y[tr])
            fold_maes.append(mean_absolute_error(y[va], m.predict(Xva)))

        self.cv_mae = float(np.mean(fold_maes))
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        self.trained = True

        imp = (1 - self.cv_mae / self.base_mae) * 100
        print(f"  ✅ CV MAE: {self.cv_mae:.1f} pts  |  Baseline MAE: {self.base_mae:.1f} pts  |  Improvement: {imp:.1f}%")

    def predict(self, feat_dict):
        row = np.array([[feat_dict.get(c, 0.0) for c in self.cols]])
        pred = float(self.model.predict(self.scaler.transform(row))[0])
        conf = round(min(0.9, max(0.0, 1 - self.cv_mae / self.base_mae)), 2)
        return {"margin": round(pred, 1), "mae": round(self.cv_mae, 1), "conf": conf}

# ── ESPN ─────────────────────────────────────────────────────────────────────
def fetch_games():
    try:
        data = requests.get(ESPN_URL, timeout=10).json()
        out  = []
        for e in data.get("events", []):
            comp  = e["competitions"][0]
            comps = comp["competitors"]
            home  = next((c for c in comps if c["homeAway"]=="home"), {})
            away  = next((c for c in comps if c["homeAway"]=="away"), {})
            st    = e["status"]["type"]

            # Parse spread
            spread, stm = None, None
            for odd in comp.get("odds", []):
                parts = odd.get("details","").split()
                if len(parts) >= 2:
                    stm = parts[0]
                    for p in parts[1:]:
                        try: spread = float(p); break
                        except ValueError: pass
                    if spread is not None: break

            sit    = comp.get("situation", {})
            period = sit.get("period") or comp.get("status",{}).get("period") or 0

            out.append({
                "id":         e.get("id"),
                "name":       e.get("shortName",""),
                "state":      st.get("state","pre"),
                "desc":       st.get("description",""),
                "home":       norm_abbr(home.get("team",{}).get("abbreviation","")),
                "home_name":  home.get("team",{}).get("displayName",""),
                "home_score": int(home.get("score") or 0),
                "home_rec":   (home.get("records") or [{}])[0].get("summary",""),
                "away":       norm_abbr(away.get("team",{}).get("abbreviation","")),
                "away_name":  away.get("team",{}).get("displayName",""),
                "away_score": int(away.get("score") or 0),
                "away_rec":   (away.get("records") or [{}])[0].get("summary",""),
                "spread":     spread,
                "spread_fav": stm,
                "clock":      st.get("displayClock","12:00"),
                "period":     period,
            })
        return out
    except Exception as ex:
        print(f"  ⚠️  ESPN error: {ex}"); return []

# ── SNAPSHOT & PREDICTION ─────────────────────────────────────────────────────
def _team_id(abbr):
    for t in nba_teams_static.get_teams():
        if t["abbreviation"].upper() == abbr.upper():
            return t["id"]
    return None

def _snapshot(abbr, feat_df):
    tid = _team_id(abbr)
    if tid is None: return None
    rows = feat_df[feat_df["TEAM_ID"] == tid].sort_values("GAME_DATE")
    if rows.empty: return None
    return rows.iloc[-1].to_dict()

def _make_features(h_snap, a_snap):
    feat = {}
    for k, v in h_snap.items():
        if k in ("TEAM_ID","GAME_ID","GAME_DATE","PTS_ACTUAL"): continue
        feat[f"H_{k}"] = float(v) if isinstance(v,(int,float)) and not (isinstance(v,float) and np.isnan(v)) else 0.0
    for k, v in a_snap.items():
        if k in ("TEAM_ID","GAME_ID","GAME_DATE","PTS_ACTUAL"): continue
        feat[f"A_{k}"] = float(v) if isinstance(v,(int,float)) and not (isinstance(v,float) and np.isnan(v)) else 0.0

    feat["H_IS_HOME"] = 1.0
    feat["A_IS_HOME"] = 0.0

    # Differentials
    for k in list(feat.keys()):
        if k.startswith("H_R") or k.startswith("H_WIN") or k.startswith("H_STREAK"):
            base = k[2:]
            hv, av = feat.get(f"H_{base}", 0), feat.get(f"A_{base}", 0)
            feat[f"D_{base}"] = hv - av

    if "H_REST_DAYS" in feat and "A_REST_DAYS" in feat:
        feat["D_REST"] = feat["H_REST_DAYS"] - feat["A_REST_DAYS"]

    return feat

def pick(game, predictor, feat_df):
    hs = _snapshot(game["home"], feat_df)
    as_ = _snapshot(game["away"], feat_df)
    if hs is None or as_ is None:
        return {"error": f"Missing data for {game['home']} or {game['away']}"}

    feats  = _make_features(hs, as_)
    result = predictor.predict(feats)
    margin = result["margin"]
    spread = game["spread"]

    rec = {
        **game,
        "predicted_margin": margin,
        "mae": result["mae"],
        "conf": result["conf"],
        "type": "pre",
    }

    if spread is not None:
        edge = margin - (-spread)   # >0 = home covers, <0 = away covers
        rec["edge"] = round(edge, 1)
        if abs(edge) < 2:
            rec["pick"], rec["strength"] = "PASS — edge too small", "WEAK"
        elif edge > 0:
            rec["pick"]     = f"{game['home']} covers {spread:+.1f}"
            rec["strength"] = "STRONG" if abs(edge) > 5 else "MODERATE"
        else:
            rec["pick"]     = f"{game['away']} covers +{abs(spread):.1f}"
            rec["strength"] = "STRONG" if abs(edge) > 5 else "MODERATE"
    else:
        rec["pick"], rec["strength"], rec["edge"] = f"No line — proj margin: {margin:+.1f}", "N/A", None

    return rec

def pick_live(game, predictor, feat_df):
    rec = pick(game, predictor, feat_df)
    if "error" in rec: return rec
    rec["type"] = "live"

    period = int(game["period"] or 1)
    try:
        parts = game["clock"].split(":")
        cmins = int(parts[0]) + int(parts[1])/60
    except: cmins = 6.0

    played    = (period-1)*12 + (12-cmins)
    remaining = max(0.0, 48.0 - played)
    pct       = remaining / 48.0
    live_mar  = game["home_score"] - game["away_score"]
    blended   = round(rec["predicted_margin"]*pct + live_mar*(1-pct), 1)

    rec["live_margin"]   = live_mar
    rec["blended"]       = blended
    rec["period"]        = period
    rec["mins_left"]     = round(remaining, 1)
    rec["pct_left"]      = round(pct*100, 1)

    spread = game["spread"]
    if spread is not None:
        le = blended - (-spread)
        rec["live_edge"] = round(le, 1)
        if abs(le) < 2:   rec["live_pick"] = "PASS — too close"
        elif le > 0:      rec["live_pick"] = f"{game['home']} covers (live)"
        else:             rec["live_pick"] = f"{game['away']} covers (live)"
    else:
        rec["live_pick"] = f"Proj final: {blended:+.1f} (home)"

    if abs(live_mar) > 15 and pct < 0.25:
        rec["alert"] = "⚠️  Big lead late — check live line on trailing team"
    elif abs(live_mar) < 3 and pct < 0.15:
        rec["alert"] = "🔥 Nail-biter in final minutes — high variance"
    else:
        rec["alert"] = ""

    return rec

# ── PRINT ─────────────────────────────────────────────────────────────────────
def print_rec(r):
    print(f"\n{'─'*54}")
    print(f"🏀  {r['away_name']} ({r['away_rec']}) @ {r['home_name']} ({r['home_rec']})")
    if r["type"] == "live":
        print(f"    Q{r['period']} | {r['mins_left']:.0f} min left | "
              f"Live: {r['away']} {r['live_margin']*-1:+.0f} / {r['home']} {r['live_margin']:+.0f}")
        print(f"    Blended proj margin (home): {r['blended']:+.1f}  |  Live edge: {r.get('live_edge','N/A')}")
        print(f"    📌 LIVE PICK:  {r['live_pick']}")
        if r.get("alert"): print(f"    {r['alert']}")
    else:
        print(f"    Predicted margin (home): {r['predicted_margin']:+.1f} pts  (model MAE ±{r['mae']} pts)")
        if r.get("spread") is not None:
            print(f"    Vegas: {r.get('spread_fav','')} {r['spread']:+.1f}  |  Edge: {r['edge']:+.1f}")
        print(f"    📌 PICK:  {r['pick']}")
        print(f"    Strength: {r['strength']}  |  Model confidence: {r['conf']:.0%}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("\n🏀 NBA SPREAD PREDICTOR — Training Pipeline")
    print("="*55)

    all_games = []
    for s in SEASONS:
        df = fetch_season_games(s)
        df["SEASON"] = s
        all_games.append(df)
        time.sleep(0.3)

    games_df = pd.concat(all_games, ignore_index=True)
    print(f"  📋 Total records: {len(games_df):,}")

    print("\n🔧 Building features (no leakage)...")
    feat_df = build_features(games_df)
    print(f"  Feature rows: {len(feat_df):,}")

    print("\n🔗 Building matchup dataset...")
    matchups = build_matchups(games_df, feat_df)
    print(f"  Matchup rows: {len(matchups):,}")

    # Sanity check — print target distribution
    m = matchups["MARGIN"]
    print(f"  Margin stats: mean={m.mean():.1f}, std={m.std():.1f}, min={m.min():.0f}, max={m.max():.0f}")

    print("\n🤖 Training XGBoost...")
    predictor = SpreadPredictor()
    predictor.fit(matchups)

    print("\n📡 Fetching ESPN games...")
    games = fetch_games()
    print(f"  🎮 Found: {len(games)} games")

    recs = []
    print("\n" + "="*55)
    print("📊 PICKS")
    print("="*55)

    for g in games:
        state = g["state"]
        if state == "in":
            r = pick_live(g, predictor, feat_df)
        elif state == "pre":
            r = pick(g, predictor, feat_df)
        else:
            continue

        if "error" in r:
            print(f"\n  ⚠️  {g['name']}: {r['error']}")
        else:
            recs.append(r)
            print_rec(r)

    print(f"\n\n✅ Done — {len(recs)} games analyzed.")
    return recs

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║          NBA SPREAD PREDICTOR  v3.0                     ║
║  ⚠️  Educational purposes only — NOT financial advice   ║
╚══════════════════════════════════════════════════════════╝""")
    main()
