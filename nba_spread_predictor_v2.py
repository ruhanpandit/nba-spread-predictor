"""
NBA Spread Predictor — ML-powered betting analysis tool
Uses nba_api for historical data + ESPN for live game data
Model: XGBoost ensemble with feature engineering

⚠️  DISCLAIMER: This tool is for educational/entertainment purposes only.
    Sports betting involves significant financial risk. This is NOT financial
    advice. Never bet more than you can afford to lose. Please gamble responsibly.
"""

import os
import json
import time
import warnings
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── NBA API imports ──────────────────────────────────────────────────────────
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams as nba_teams_static

# ── ML imports ───────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


# ════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════════════

SEASONS = ["2022-23", "2023-24", "2024-25"]
ROLLING_WINDOWS = [5, 10, 20]
REQUEST_DELAY = 0.7
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".nba_cache")

os.makedirs(CACHE_DIR, exist_ok=True)

# ── IMPORTANT: PLUS_MINUS removed — it directly encodes the game margin
# and causes severe data leakage. Only use stats that don't reveal the outcome.
STAT_COLS = [
    "PTS", "AST", "REB", "FG_PCT", "FG3_PCT", "FT_PCT",
    "TOV", "STL", "BLK", "OREB", "DREB",
]


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _cache_path(key: str) -> str:
    safe = key.replace("/", "_").replace(" ", "_")
    return os.path.join(CACHE_DIR, f"{safe}.json")


def _load_cache(key: str):
    p = _cache_path(key)
    if os.path.exists(p):
        age = time.time() - os.path.getmtime(p)
        if age < 3600 * 6:
            with open(p) as f:
                return json.load(f)
    return None


def _save_cache(key: str, data):
    with open(_cache_path(key), "w") as f:
        json.dump(data, f)


def _get_all_teams() -> dict:
    all_teams = nba_teams_static.get_teams()
    mapping = {}
    for t in all_teams:
        mapping[t["abbreviation"]] = t["id"]
        mapping[t["full_name"].upper()] = t["id"]
        mapping[str(t["id"])] = t["id"]
    return mapping


TEAM_MAP = _get_all_teams()


# ════════════════════════════════════════════════════════════════════════════
#  DATA COLLECTION
# ════════════════════════════════════════════════════════════════════════════

def fetch_season_games(season: str) -> pd.DataFrame:
    cache_key = f"season_games_{season}"
    cached = _load_cache(cache_key)
    if cached:
        return pd.DataFrame(cached)

    print(f"  📥 Fetching games for {season}...")
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",
        season_type_nullable="Regular Season",
    )
    time.sleep(REQUEST_DELAY)
    df = finder.get_data_frames()[0]
    _save_cache(cache_key, df.to_dict("records"))
    return df


# ════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════

def build_team_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team/game, compute rolling stats using ONLY past games (strict lag).
    PLUS_MINUS is excluded — it encodes the outcome and causes leakage.
    """
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    feature_rows = []

    for team_id_val, grp in df.groupby("TEAM_ID"):
        grp = grp.sort_values("GAME_DATE").reset_index(drop=True)
        available_cols = [c for c in STAT_COLS if c in grp.columns]

        for i in range(len(grp)):
            row = grp.iloc[i]
            past = grp.iloc[:i]  # strictly past — no current game included

            feat = {
                "TEAM_ID": team_id_val,
                "GAME_ID": row["GAME_ID"],
                "GAME_DATE": row["GAME_DATE"],
                "HOME": 1 if "vs." in str(row.get("MATCHUP", "")) else 0,
                "PTS_ACTUAL": row.get("PTS", np.nan),
                "GAMES_PLAYED": i,
            }

            for w in ROLLING_WINDOWS:
                window = past.tail(w)
                for col in available_cols:
                    vals = window[col].dropna()
                    prefix = f"ROLL{w}_{col}"
                    feat[f"{prefix}_MEAN"] = float(vals.mean()) if len(vals) > 0 else np.nan
                    feat[f"{prefix}_STD"]  = float(vals.std())  if len(vals) > 1 else 0.0

            # Home vs away splits (last 10 games)
            if len(past) > 0:
                past_home = past[past["MATCHUP"].str.contains("vs\\.", na=False)]
                past_away = past[past["MATCHUP"].str.contains("@", na=False)]
                for col in ["PTS", "FG_PCT"]:
                    if col in grp.columns:
                        feat[f"HOME_SPLIT_{col}"] = float(past_home[col].tail(10).mean()) if len(past_home) > 0 else np.nan
                        feat[f"AWAY_SPLIT_{col}"] = float(past_away[col].tail(10).mean()) if len(past_away) > 0 else np.nan

            # Win/loss streak
            if "WL" in grp.columns and len(past) > 0:
                recent_wl = past["WL"].tail(10).tolist()
                streak = 0
                last = recent_wl[-1] if recent_wl else "W"
                for r in reversed(recent_wl):
                    if r == last:
                        streak += 1
                    else:
                        break
                feat["STREAK"] = streak if last == "W" else -streak
                feat["WIN_RATE_L10"] = sum(1 for r in recent_wl if r == "W") / max(len(recent_wl), 1)
            else:
                feat["STREAK"] = 0
                feat["WIN_RATE_L10"] = 0.5

            # Rest days
            if i > 0:
                feat["REST_DAYS"] = int((row["GAME_DATE"] - grp.iloc[i - 1]["GAME_DATE"]).days)
            else:
                feat["REST_DAYS"] = 3

            feature_rows.append(feat)

    result = pd.DataFrame(feature_rows)
    result = result.dropna(thresh=len(result.columns) // 2)
    return result


def build_matchup_dataset(games_df: pd.DataFrame, rolling_df: pd.DataFrame) -> pd.DataFrame:
    games_df = games_df.copy()
    games_df["GAME_DATE"] = pd.to_datetime(games_df["GAME_DATE"])
    rolling_df = rolling_df.copy()
    rolling_df["GAME_DATE"] = pd.to_datetime(rolling_df["GAME_DATE"])

    home_games = games_df[games_df["MATCHUP"].str.contains("vs\\.", na=False)].copy()
    away_games = games_df[games_df["MATCHUP"].str.contains("@", na=False)].copy()

    home_merged = home_games.merge(rolling_df, on=["TEAM_ID", "GAME_ID", "GAME_DATE"], how="inner")
    away_merged = away_games.merge(rolling_df, on=["TEAM_ID", "GAME_ID", "GAME_DATE"], how="inner")

    skip = {"GAME_ID", "GAME_DATE"}
    home_merged = home_merged.rename(columns={c: f"HOME_{c}" for c in home_merged.columns if c not in skip})
    away_merged = away_merged.rename(columns={c: f"AWAY_{c}" for c in away_merged.columns if c not in skip})

    matchups = home_merged.merge(away_merged, on=["GAME_ID", "GAME_DATE"], how="inner")
    matchups["ACTUAL_MARGIN"] = matchups["HOME_PTS_ACTUAL"] - matchups["AWAY_PTS_ACTUAL"]

    # Differential features — most predictive signals
    for col in STAT_COLS:
        for w in ROLLING_WINDOWS:
            h = f"HOME_ROLL{w}_{col}_MEAN"
            a = f"AWAY_ROLL{w}_{col}_MEAN"
            if h in matchups.columns and a in matchups.columns:
                matchups[f"DIFF{w}_{col}"] = matchups[h] - matchups[a]

    if "HOME_STREAK" in matchups.columns and "AWAY_STREAK" in matchups.columns:
        matchups["STREAK_DIFF"] = matchups["HOME_STREAK"] - matchups["AWAY_STREAK"]
    if "HOME_REST_DAYS" in matchups.columns and "AWAY_REST_DAYS" in matchups.columns:
        matchups["REST_DIFF"] = matchups["HOME_REST_DAYS"] - matchups["AWAY_REST_DAYS"]
    if "HOME_WIN_RATE_L10" in matchups.columns and "AWAY_WIN_RATE_L10" in matchups.columns:
        matchups["WIN_RATE_DIFF"] = matchups["HOME_WIN_RATE_L10"] - matchups["AWAY_WIN_RATE_L10"]

    matchups = matchups.dropna(subset=["ACTUAL_MARGIN"])
    return matchups


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {
        "GAME_ID", "GAME_DATE", "ACTUAL_MARGIN",
        "HOME_PTS_ACTUAL", "AWAY_PTS_ACTUAL",
        "HOME_TEAM_ID", "AWAY_TEAM_ID",
        "HOME_GAME_ID", "AWAY_GAME_ID",
        "HOME_GAME_DATE", "AWAY_GAME_DATE",
        "HOME_MATCHUP", "AWAY_MATCHUP",
        "HOME_SEASON", "AWAY_SEASON",
        "HOME_WL", "AWAY_WL",
    }
    return [
        c for c in df.columns
        if c not in exclude
        and df[c].dtype in [np.float64, np.int64, float, int]
        and not c.startswith("HOME_TEAM_")
        and not c.startswith("AWAY_TEAM_")
    ]


# ════════════════════════════════════════════════════════════════════════════
#  MODEL
# ════════════════════════════════════════════════════════════════════════════

class SpreadPredictor:
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.75,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=2.0,
            reg_lambda=5.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_trained = False
        self.cv_mae = None
        self.baseline_mae = None

    def fit(self, matchups_df: pd.DataFrame):
        feat_cols = get_feature_cols(matchups_df)
        self.feature_cols = feat_cols

        X = matchups_df[feat_cols].fillna(0).values
        y = matchups_df["ACTUAL_MARGIN"].values

        self.baseline_mae = float(np.mean(np.abs(y)))

        tscv = TimeSeriesSplit(n_splits=5)
        maes = []
        for train_idx, val_idx in tscv.split(X):
            Xtr, Xval = X[train_idx], X[val_idx]
            ytr, yval = y[train_idx], y[val_idx]
            scaler = StandardScaler()
            Xtr_s  = scaler.fit_transform(Xtr)
            Xval_s = scaler.transform(Xval)
            m = XGBRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.03,
                subsample=0.75, colsample_bytree=0.7, min_child_weight=5,
                reg_alpha=2.0, reg_lambda=5.0, random_state=42,
                n_jobs=-1, verbosity=0,
            )
            m.fit(Xtr_s, ytr)
            preds = m.predict(Xval_s)
            maes.append(mean_absolute_error(yval, preds))

        self.cv_mae = float(np.mean(maes))

        X_s = self.scaler.fit_transform(X)
        self.model.fit(X_s, y)

        self.is_trained = True
        improvement = (1 - self.cv_mae / self.baseline_mae) * 100
        print(f"  ✅ Model trained | CV MAE: {self.cv_mae:.2f} pts  |  Baseline: {self.baseline_mae:.2f} pts  |  Improvement: {improvement:.1f}%")

    def predict(self, feature_dict: dict) -> dict:
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")

        row = np.array([[feature_dict.get(c, 0.0) for c in self.feature_cols]])
        row_s = self.scaler.transform(row)
        predicted = float(self.model.predict(row_s)[0])

        improvement = max(0.0, 1.0 - self.cv_mae / self.baseline_mae)
        confidence = round(min(0.95, improvement), 2)

        return {
            "predicted_margin": round(predicted, 1),
            "model_mae": round(self.cv_mae, 2),
            "confidence": confidence,
        }


# ════════════════════════════════════════════════════════════════════════════
#  ESPN LIVE DATA
# ════════════════════════════════════════════════════════════════════════════

def fetch_espn_scoreboard() -> list:
    try:
        resp = requests.get(ESPN_SCOREBOARD, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        events = data.get("events", [])
        games = []
        for e in events:
            comps = e.get("competitions", [{}])[0]
            competitors = comps.get("competitors", [])
            home = next((c for c in competitors if c["homeAway"] == "home"), {})
            away = next((c for c in competitors if c["homeAway"] == "away"), {})
            status = e.get("status", {}).get("type", {})

            odds_list = comps.get("odds", [{}])
            spread_val = None
            spread_team = None
            if odds_list:
                details = odds_list[0].get("details", "")
                parts = details.split()
                if len(parts) >= 2:
                    spread_team = parts[0]
                    for p in parts[1:]:
                        try:
                            spread_val = float(p)
                            break
                        except ValueError:
                            continue

            situation = comps.get("situation", {})
            period = situation.get("period") or comps.get("status", {}).get("period") or 0

            game = {
                "game_id": e.get("id"),
                "short_name": e.get("shortName", ""),
                "status": status.get("description", ""),
                "state": status.get("state", "pre"),
                "home_team": home.get("team", {}).get("abbreviation", ""),
                "home_team_name": home.get("team", {}).get("displayName", ""),
                "home_score": int(home.get("score") or 0),
                "home_record": (home.get("records") or [{}])[0].get("summary", ""),
                "away_team": away.get("team", {}).get("abbreviation", ""),
                "away_team_name": away.get("team", {}).get("displayName", ""),
                "away_score": int(away.get("score") or 0),
                "away_record": (away.get("records") or [{}])[0].get("summary", ""),
                "spread": spread_val,
                "spread_team": spread_team,
                "clock": status.get("displayClock", "12:00"),
                "period": period,
            }
            games.append(game)
        return games
    except Exception as ex:
        print(f"  ⚠️  ESPN fetch error: {ex}")
        return []


# ════════════════════════════════════════════════════════════════════════════
#  RECOMMENDATION ENGINE
# ════════════════════════════════════════════════════════════════════════════

def _get_team_snapshot(team_abbr: str, rolling_df: pd.DataFrame) -> dict:
    all_teams = nba_teams_static.get_teams()
    tid = None
    for t in all_teams:
        if t["abbreviation"].upper() == team_abbr.upper():
            tid = t["id"]
            break
    if tid is None:
        return {}

    rows = rolling_df[rolling_df["TEAM_ID"] == tid].sort_values("GAME_DATE")
    if rows.empty:
        return {}
    return rows.iloc[-1].to_dict()


def _build_matchup_features(home_snap: dict, away_snap: dict) -> dict:
    combined = {}

    for k, v in home_snap.items():
        if k not in ["TEAM_ID", "GAME_ID", "GAME_DATE", "PTS_ACTUAL"]:
            combined[f"HOME_{k}"] = v if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)) else 0.0

    for k, v in away_snap.items():
        if k not in ["TEAM_ID", "GAME_ID", "GAME_DATE", "PTS_ACTUAL"]:
            combined[f"AWAY_{k}"] = v if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)) else 0.0

    combined["HOME_HOME"] = 1
    combined["AWAY_HOME"] = 0

    for col in STAT_COLS:
        for w in ROLLING_WINDOWS:
            h_key = f"HOME_ROLL{w}_{col}_MEAN"
            a_key = f"AWAY_ROLL{w}_{col}_MEAN"
            if h_key in combined and a_key in combined:
                combined[f"DIFF{w}_{col}"] = combined[h_key] - combined[a_key]

    if "HOME_STREAK" in combined and "AWAY_STREAK" in combined:
        combined["STREAK_DIFF"] = combined["HOME_STREAK"] - combined["AWAY_STREAK"]
    if "HOME_REST_DAYS" in combined and "AWAY_REST_DAYS" in combined:
        combined["REST_DIFF"] = combined["HOME_REST_DAYS"] - combined["AWAY_REST_DAYS"]
    if "HOME_WIN_RATE_L10" in combined and "AWAY_WIN_RATE_L10" in combined:
        combined["WIN_RATE_DIFF"] = combined["HOME_WIN_RATE_L10"] - combined["AWAY_WIN_RATE_L10"]

    return combined


def generate_pregame_recommendation(game: dict, predictor: SpreadPredictor,
                                     rolling_df: pd.DataFrame) -> dict:
    home_snap = _get_team_snapshot(game["home_team"], rolling_df)
    away_snap = _get_team_snapshot(game["away_team"], rolling_df)

    if not home_snap or not away_snap:
        return {"error": f"No historical data for {game['home_team']} or {game['away_team']}"}

    features = _build_matchup_features(home_snap, away_snap)
    result = predictor.predict(features)
    margin = result["predicted_margin"]
    spread = game.get("spread")

    rec = {
        "game": game["short_name"],
        "home": game["home_team_name"],
        "away": game["away_team_name"],
        "home_abbr": game["home_team"],
        "away_abbr": game["away_team"],
        "home_record": game.get("home_record", ""),
        "away_record": game.get("away_record", ""),
        "predicted_margin": margin,
        "spread": spread,
        "spread_team": game.get("spread_team"),
        "confidence": result["confidence"],
        "model_mae": result["model_mae"],
        "type": "pregame",
    }

    if spread is not None:
        edge = margin - (-spread)
        rec["edge"] = round(edge, 1)
        if abs(edge) < 2.0:
            rec["pick"] = "PASS — edge too small"
            rec["pick_team"] = None
            rec["strength"] = "WEAK"
        elif edge > 0:
            rec["pick"] = f"{game['home_team']} covers {spread:+.1f}"
            rec["pick_team"] = game["home_team"]
            rec["strength"] = "STRONG" if abs(edge) > 5 else "MODERATE"
        else:
            rec["pick"] = f"{game['away_team']} covers +{abs(spread):.1f}"
            rec["pick_team"] = game["away_team"]
            rec["strength"] = "STRONG" if abs(edge) > 5 else "MODERATE"
    else:
        rec["pick"] = f"No line available — projected home margin: {margin:+.1f}"
        rec["pick_team"] = None
        rec["strength"] = "N/A"
        rec["edge"] = None

    return rec


def generate_ingame_recommendation(game: dict, predictor: SpreadPredictor,
                                    rolling_df: pd.DataFrame) -> dict:
    base = generate_pregame_recommendation(game, predictor, rolling_df)
    if "error" in base:
        return base

    base["type"] = "ingame"
    period = int(game.get("period") or 1)
    clock  = str(game.get("clock") or "12:00")
    home_score = int(game.get("home_score") or 0)
    away_score = int(game.get("away_score") or 0)
    live_margin = home_score - away_score

    try:
        parts = clock.split(":")
        clock_mins = int(parts[0]) + int(parts[1]) / 60
    except Exception:
        clock_mins = 6.0

    mins_played = (period - 1) * 12 + (12 - clock_mins)
    mins_remaining = max(0.0, 48.0 - mins_played)
    pct_remaining = mins_remaining / 48.0

    blended = round(base["predicted_margin"] * pct_remaining + live_margin * (1 - pct_remaining), 1)
    base["predicted_final_margin"] = blended
    base["live_margin"] = live_margin
    base["period"] = period
    base["mins_remaining"] = round(mins_remaining, 1)
    base["pct_remaining"] = round(pct_remaining * 100, 1)

    spread = game.get("spread")
    if spread is not None:
        live_edge = blended - (-spread)
        base["live_edge"] = round(live_edge, 1)
        if abs(live_edge) < 2:
            base["ingame_pick"] = "PASS — too close to live line"
        elif live_edge > 0:
            base["ingame_pick"] = f"{game['home_team']} covers (live)"
        else:
            base["ingame_pick"] = f"{game['away_team']} covers (live)"
    else:
        base["ingame_pick"] = f"Projected final margin: {blended:+.1f} (home)"

    if abs(live_margin) > 15 and pct_remaining < 0.25:
        base["alert"] = "⚠️  Large lead late — trailing team may have live value if model agrees"
    elif abs(live_margin) < 3 and pct_remaining < 0.15:
        base["alert"] = "🔥 Tight game in final minutes — high variance, be cautious"
    else:
        base["alert"] = ""

    return base


# ════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def train_model(seasons=SEASONS):
    print("\n🏀 NBA SPREAD PREDICTOR — Training Pipeline")
    print("=" * 55)

    all_games = []
    for season in seasons:
        df = fetch_season_games(season)
        df["SEASON"] = season
        all_games.append(df)
        time.sleep(0.3)

    games_df = pd.concat(all_games, ignore_index=True)
    print(f"  📋 Total game records: {len(games_df):,}")

    print("\n🔧 Engineering features...")
    rolling_df = build_team_rolling_features(games_df)

    print("\n🔗 Building matchup dataset...")
    matchups = build_matchup_dataset(games_df, rolling_df)
    print(f"  Matchup rows: {len(matchups):,}")

    print("\n🤖 Training model...")
    predictor = SpreadPredictor()
    predictor.fit(matchups)

    return predictor, rolling_df, games_df, matchups


def run_analysis():
    predictor, rolling_df, games_df, matchups = train_model()

    print("\n📡 Fetching live ESPN data...")
    today_games = fetch_espn_scoreboard()
    print(f"  🎮 Games found: {len(today_games)}")

    if not today_games:
        print("  No games scheduled today.")
        return []

    recommendations = []
    print("\n" + "=" * 55)
    print("📊 SPREAD RECOMMENDATIONS")
    print("=" * 55)

    for game in today_games:
        state = game.get("state", "pre")
        if state == "in":
            rec = generate_ingame_recommendation(game, predictor, rolling_df)
        elif state == "pre":
            rec = generate_pregame_recommendation(game, predictor, rolling_df)
        else:
            continue

        if "error" not in rec:
            recommendations.append(rec)
            _print_rec(rec)
        else:
            print(f"\n  ⚠️  {game['short_name']}: {rec['error']}")

    return recommendations


def _print_rec(rec: dict):
    print(f"\n{'─' * 52}")
    print(f"🏀  {rec['away']} ({rec.get('away_record','')}) @ {rec['home']} ({rec.get('home_record','')})")

    if rec["type"] == "ingame":
        print(f"    Q{rec['period']} | {rec['mins_remaining']:.0f} min remaining | Live: {rec['away_abbr']} {rec['live_margin']*-1:+.0f} / {rec['home_abbr']} {rec['live_margin']:+.0f}")
        print(f"    Projected final margin (home): {rec['predicted_final_margin']:+.1f}")
        print(f"    📌 IN-GAME PICK: {rec.get('ingame_pick', 'N/A')}")
        if rec.get("alert"):
            print(f"    {rec['alert']}")
    else:
        print(f"    Model predicts home margin: {rec['predicted_margin']:+.1f} pts  (±{rec['model_mae']} MAE)")
        if rec.get("spread") is not None:
            print(f"    Vegas spread: {rec.get('spread_team','')} {rec['spread']:+.1f}  |  Model edge: {rec.get('edge', 0):+.1f}")
        print(f"    📌 PICK:  {rec['pick']}")
        print(f"    Strength: {rec['strength']}  |  Confidence vs baseline: {rec['confidence']:.0%}")


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║          NBA SPREAD PREDICTOR  v2.0                     ║
║  ⚠️  Educational purposes only — NOT financial advice   ║
╚══════════════════════════════════════════════════════════╝
""")
    recs = run_analysis()
    print(f"\n\n✅ Done — {len(recs)} games analyzed.")
    print("   Run again closer to tip-off for fresher lines.")
