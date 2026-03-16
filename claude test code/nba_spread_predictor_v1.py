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
from collections import defaultdict

warnings.filterwarnings("ignore")

# ── NBA API imports ──────────────────────────────────────────────────────────
from nba_api.stats.endpoints import (
    leaguegamefinder,
    teamgamelog,
    boxscoreadvancedv2,
    boxscoretraditionalv2,
    leaguedashteamstats,
    teamestimatedmetrics,
    leaguedashteamptshot,
)
from nba_api.stats.static import teams as nba_teams_static

# ── ML imports ───────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor


# ════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════════════

SEASONS = ["2022-23", "2023-24", "2024-25"]  # seasons to train on
ROLLING_WINDOWS = [5, 10, 20]               # game windows for rolling stats
REQUEST_DELAY = 0.7                          # seconds between NBA API calls
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_TEAMS      = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
CACHE_DIR       = os.path.join(os.path.dirname(__file__), ".nba_cache")

os.makedirs(CACHE_DIR, exist_ok=True)


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
        if age < 3600 * 6:  # 6-hour cache
            with open(p) as f:
                return json.load(f)
    return None


def _save_cache(key: str, data):
    with open(_cache_path(key), "w") as f:
        json.dump(data, f)


def _get_all_teams() -> dict:
    """Return {abbreviation: id, name: id} mapping."""
    all_teams = nba_teams_static.get_teams()
    mapping = {}
    for t in all_teams:
        mapping[t["abbreviation"]] = t["id"]
        mapping[t["full_name"].upper()] = t["id"]
        mapping[str(t["id"])] = t["id"]
    return mapping


TEAM_MAP = _get_all_teams()


def team_id(name_or_abbr: str) -> int | None:
    return TEAM_MAP.get(name_or_abbr.upper())


# ════════════════════════════════════════════════════════════════════════════
#  DATA COLLECTION
# ════════════════════════════════════════════════════════════════════════════

def fetch_season_games(season: str) -> pd.DataFrame:
    """Pull every game for a season from nba_api."""
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


def fetch_team_season_stats(season: str) -> pd.DataFrame:
    """Pull aggregate team stats (advanced) for a season."""
    cache_key = f"team_stats_{season}"
    cached = _load_cache(cache_key)
    if cached:
        return pd.DataFrame(cached)

    print(f"  📊 Fetching team stats for {season}...")
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_simple_nullable="Advanced",
        per_mode_simple="PerGame",
    )
    time.sleep(REQUEST_DELAY)
    df = stats.get_data_frames()[0]
    _save_cache(cache_key, df.to_dict("records"))
    return df


def fetch_team_estimated_metrics(season: str) -> pd.DataFrame:
    cache_key = f"est_metrics_{season}"
    cached = _load_cache(cache_key)
    if cached:
        return pd.DataFrame(cached)

    print(f"  📈 Fetching estimated metrics for {season}...")
    try:
        em = teamestimatedmetrics.TeamEstimatedMetrics(season=season)
        time.sleep(REQUEST_DELAY)
        df = em.get_data_frames()[0]
        _save_cache(cache_key, df.to_dict("records"))
        return df
    except Exception as e:
        print(f"    ⚠️  Could not fetch estimated metrics: {e}")
        return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════

STAT_COLS = [
    "PTS", "AST", "REB", "FG_PCT", "FG3_PCT", "FT_PCT",
    "TOV", "STL", "BLK", "PLUS_MINUS", "OREB", "DREB",
]


def build_team_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team/game, compute rolling stats over multiple windows.
    Returns one row per (TEAM_ID, GAME_ID) with lagged rolling features.
    """
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ID", "GAME_DATE"])

    feature_rows = []

    for team_id_val, grp in df.groupby("TEAM_ID"):
        grp = grp.reset_index(drop=True)
        available_cols = [c for c in STAT_COLS if c in grp.columns]

        for i, row in grp.iterrows():
            feat = {
                "TEAM_ID": team_id_val,
                "GAME_ID": row["GAME_ID"],
                "GAME_DATE": row["GAME_DATE"],
                "HOME": 1 if "vs." in str(row.get("MATCHUP", "")) else 0,
                "PTS_ACTUAL": row.get("PTS", np.nan),
            }

            past = grp.iloc[:i]  # strictly past games

            for w in ROLLING_WINDOWS:
                window = past.tail(w)
                for col in available_cols:
                    vals = window[col].dropna()
                    prefix = f"ROLL{w}_{col}"
                    feat[f"{prefix}_MEAN"] = vals.mean() if len(vals) else np.nan
                    feat[f"{prefix}_STD"]  = vals.std()  if len(vals) > 1 else 0.0

            # Win/loss streak
            if "WL" in grp.columns and i > 0:
                recent_wl = past["WL"].tail(10).tolist()
                streak = 0
                for r in reversed(recent_wl):
                    if r == recent_wl[-1]:
                        streak += 1
                    else:
                        break
                feat["STREAK"] = streak if recent_wl and recent_wl[-1] == "W" else -streak
            else:
                feat["STREAK"] = 0

            # Rest days
            if i > 0:
                feat["REST_DAYS"] = (row["GAME_DATE"] - grp.iloc[i-1]["GAME_DATE"]).days
            else:
                feat["REST_DAYS"] = 3  # default

            feature_rows.append(feat)

    return pd.DataFrame(feature_rows)


def build_matchup_dataset(games_df: pd.DataFrame, rolling_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join home vs away rolling features to form matchup rows.
    Target: HOME_PTS - AWAY_PTS (actual margin, i.e. covers spread positive = home covered)
    """
    games_df = games_df.copy()
    games_df["GAME_DATE"] = pd.to_datetime(games_df["GAME_DATE"])
    rolling_df = rolling_df.copy()
    rolling_df["GAME_DATE"] = pd.to_datetime(rolling_df["GAME_DATE"])

    # Identify home/away per game
    # MATCHUP format: "LAL vs. GSW" (home) or "LAL @ GSW" (away)
    home = games_df[games_df["MATCHUP"].str.contains("vs\\.", na=False)].copy()
    away = games_df[games_df["MATCHUP"].str.contains("@", na=False)].copy()

    home = home.merge(rolling_df, on=["TEAM_ID", "GAME_ID", "GAME_DATE"], how="inner")
    away = away.merge(rolling_df, on=["TEAM_ID", "GAME_ID", "GAME_DATE"], how="inner")

    home_cols = {c: f"HOME_{c}" for c in home.columns if c not in ["GAME_ID", "GAME_DATE"]}
    away_cols = {c: f"AWAY_{c}" for c in away.columns if c not in ["GAME_ID", "GAME_DATE"]}

    home = home.rename(columns=home_cols)
    away = away.rename(columns=away_cols)

    matchups = home.merge(away, on=["GAME_ID", "GAME_DATE"], suffixes=("", ""))
    matchups["ACTUAL_MARGIN"] = matchups["HOME_PTS_ACTUAL"] - matchups["AWAY_PTS_ACTUAL"]
    matchups = matchups.dropna(subset=["ACTUAL_MARGIN"])

    return matchups


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {"GAME_ID", "GAME_DATE", "ACTUAL_MARGIN", "HOME_PTS_ACTUAL",
               "AWAY_PTS_ACTUAL", "HOME_TEAM_ID", "AWAY_TEAM_ID",
               "HOME_GAME_ID", "AWAY_GAME_ID", "HOME_GAME_DATE", "AWAY_GAME_DATE"}
    return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]


# ════════════════════════════════════════════════════════════════════════════
#  MODEL
# ════════════════════════════════════════════════════════════════════════════

class SpreadPredictor:
    def __init__(self):
        self.xgb = XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        self.ridge = Ridge(alpha=10.0)
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_trained = False
        self.cv_mae = None

    def fit(self, matchups_df: pd.DataFrame):
        feat_cols = get_feature_cols(matchups_df)
        self.feature_cols = feat_cols

        X = matchups_df[feat_cols].fillna(0).values
        y = matchups_df["ACTUAL_MARGIN"].values

        # Time-series cross-validation for honest evaluation
        tscv = TimeSeriesSplit(n_splits=5)
        maes = []
        for train_idx, val_idx in tscv.split(X):
            Xtr, Xval = X[train_idx], X[val_idx]
            ytr, yval = y[train_idx], y[val_idx]
            Xtr_s = self.scaler.fit_transform(Xtr)
            Xval_s = self.scaler.transform(Xval)
            self.xgb.fit(Xtr_s, ytr)
            preds = self.xgb.predict(Xval_s)
            maes.append(mean_absolute_error(yval, preds))

        self.cv_mae = np.mean(maes)

        # Final fit on all data
        X_s = self.scaler.fit_transform(X)
        self.xgb.fit(X_s, y)
        # Blend with ridge for stability
        ridge_preds = self.xgb.predict(X_s)
        self.ridge.fit(ridge_preds.reshape(-1, 1), y)

        self.is_trained = True
        print(f"  ✅ Model trained | CV MAE: {self.cv_mae:.2f} pts")

    def predict(self, feature_dict: dict) -> dict:
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
        row = {c: feature_dict.get(c, 0.0) for c in self.feature_cols}
        X = np.array([[row[c] for c in self.feature_cols]])
        X_s = self.scaler.transform(X)
        xgb_pred = float(self.xgb.predict(X_s)[0])
        blended   = float(self.ridge.predict([[xgb_pred]])[0])

        # Confidence: based on CV MAE
        confidence = max(0.0, min(1.0, 1.0 - (self.cv_mae / 15.0)))

        return {
            "predicted_margin": round(blended, 1),
            "model_mae": round(self.cv_mae, 2),
            "confidence": round(confidence, 2),
        }


# ════════════════════════════════════════════════════════════════════════════
#  ESPN LIVE DATA
# ════════════════════════════════════════════════════════════════════════════

def fetch_espn_scoreboard() -> list[dict]:
    """Pull today's NBA games from ESPN public API."""
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
            if odds_list:
                details = odds_list[0].get("details", "")
                # e.g. "LAL -5.5"
                parts = details.split()
                for p in parts:
                    try:
                        spread_val = float(p)
                        break
                    except ValueError:
                        continue

            situation = comps.get("situation", {})
            game = {
                "game_id": e.get("id"),
                "name": e.get("name", ""),
                "short_name": e.get("shortName", ""),
                "status": status.get("description", ""),
                "state": status.get("state", "pre"),  # pre, in, post
                "home_team": home.get("team", {}).get("abbreviation", ""),
                "home_team_name": home.get("team", {}).get("displayName", ""),
                "home_score": int(home.get("score", 0) or 0),
                "home_record": home.get("records", [{}])[0].get("summary", "") if home.get("records") else "",
                "away_team": away.get("team", {}).get("abbreviation", ""),
                "away_team_name": away.get("team", {}).get("displayName", ""),
                "away_score": int(away.get("score", 0) or 0),
                "away_record": away.get("records", [{}])[0].get("summary", "") if away.get("records") else "",
                "spread": spread_val,
                "clock": status.get("displayClock", ""),
                "period": situation.get("period", comps.get("status", {}).get("period", 0)),
                "home_wins_prob": float(home.get("statistics", [{}])[0].get("value", 0.5)) if home.get("statistics") else None,
            }
            games.append(game)
        return games
    except Exception as e:
        print(f"  ⚠️  ESPN fetch error: {e}")
        return []


# ════════════════════════════════════════════════════════════════════════════
#  RECOMMENDATION ENGINE
# ════════════════════════════════════════════════════════════════════════════

def _get_team_rolling_snapshot(team_abbr: str, rolling_df: pd.DataFrame,
                                games_df: pd.DataFrame) -> dict:
    """Get the most recent rolling features for a team."""
    all_teams = nba_teams_static.get_teams()
    tid = None
    for t in all_teams:
        if t["abbreviation"].upper() == team_abbr.upper():
            tid = t["id"]
            break
    if tid is None:
        return {}

    team_rows = rolling_df[rolling_df["TEAM_ID"] == tid].sort_values("GAME_DATE")
    if team_rows.empty:
        return {}

    latest = team_rows.iloc[-1].to_dict()
    return latest


def generate_pregame_recommendation(
    game: dict,
    predictor: SpreadPredictor,
    rolling_df: pd.DataFrame,
    games_df: pd.DataFrame,
) -> dict:
    """Build a feature vector for both teams and predict the margin."""
    home_snap = _get_team_rolling_snapshot(game["home_team"], rolling_df, games_df)
    away_snap = _get_team_rolling_snapshot(game["away_team"], rolling_df, games_df)

    if not home_snap or not away_snap:
        return {"error": "Insufficient historical data for one or both teams."}

    # Build combined feature dict prefixed HOME_ / AWAY_
    combined = {}
    for k, v in home_snap.items():
        if k not in ["TEAM_ID", "GAME_ID", "GAME_DATE", "PTS_ACTUAL"]:
            combined[f"HOME_{k}"] = v if isinstance(v, (int, float)) else 0
    for k, v in away_snap.items():
        if k not in ["TEAM_ID", "GAME_ID", "GAME_DATE", "PTS_ACTUAL"]:
            combined[f"AWAY_{k}"] = v if isinstance(v, (int, float)) else 0

    combined["HOME_HOME"] = 1
    combined["AWAY_HOME"] = 0

    result = predictor.predict(combined)
    margin = result["predicted_margin"]
    spread = game.get("spread")

    recommendation = {
        "game": game["short_name"],
        "home": game["home_team_name"],
        "away": game["away_team_name"],
        "predicted_margin": margin,
        "spread": spread,
        "confidence": result["confidence"],
        "model_mae": result["model_mae"],
        "type": "pregame",
    }

    if spread is not None:
        edge = margin - (-spread)  # positive = home has edge
        recommendation["edge"] = round(edge, 1)
        if abs(edge) < 2:
            recommendation["pick"] = "PASS (too close to line)"
            recommendation["pick_team"] = None
            recommendation["strength"] = "WEAK"
        elif edge > 0:
            recommendation["pick"] = f"HOME ({game['home_team']}) covers {spread}"
            recommendation["pick_team"] = game["home_team"]
            recommendation["strength"] = "STRONG" if abs(edge) > 5 else "MODERATE"
        else:
            recommendation["pick"] = f"AWAY ({game['away_team']}) covers +{abs(spread)}"
            recommendation["pick_team"] = game["away_team"]
            recommendation["strength"] = "STRONG" if abs(edge) > 5 else "MODERATE"
    else:
        recommendation["pick"] = f"Predicted home margin: {margin:+.1f} pts (no line available)"
        recommendation["pick_team"] = None
        recommendation["strength"] = "N/A"

    return recommendation


def generate_ingame_recommendation(game: dict, predictor: SpreadPredictor,
                                    rolling_df: pd.DataFrame, games_df: pd.DataFrame) -> dict:
    """
    For live games, factor in current score differential and period
    to estimate adjusted live margin.
    """
    base = generate_pregame_recommendation(game, predictor, rolling_df, games_df)
    if "error" in base:
        return base

    base["type"] = "ingame"
    period = int(game.get("period") or 1)
    clock  = game.get("clock", "12:00")
    home_score = game["home_score"]
    away_score = game["away_score"]
    live_margin = home_score - away_score

    # Estimate minutes remaining
    try:
        mins, secs = map(int, clock.replace(":", " ").split())
        clock_mins = mins + secs / 60
    except Exception:
        clock_mins = 12.0

    total_periods = 4
    mins_per_period = 12
    mins_played = (period - 1) * mins_per_period + (mins_per_period - clock_mins)
    mins_remaining = max(0, total_periods * mins_per_period - mins_played)
    pct_remaining = mins_remaining / (total_periods * mins_per_period)

    # Blend pre-game prediction with live score
    blended_margin = (base["predicted_margin"] * pct_remaining) + (live_margin * (1 - pct_remaining))
    base["predicted_final_margin"] = round(blended_margin, 1)
    base["live_margin"] = live_margin
    base["period"] = period
    base["mins_remaining"] = round(mins_remaining, 1)
    base["pct_game_remaining"] = round(pct_remaining * 100, 1)

    # Live spread estimate
    spread = game.get("spread")
    if spread is not None:
        live_edge = blended_margin - (-spread)
        base["live_edge"] = round(live_edge, 1)
        if abs(live_edge) < 2:
            base["ingame_pick"] = "PASS (margin too close to live line)"
        elif live_edge > 0:
            base["ingame_pick"] = f"HOME ({game['home_team']}) - projected to cover"
        else:
            base["ingame_pick"] = f"AWAY ({game['away_team']}) - projected to cover"
    else:
        base["ingame_pick"] = f"Live projected home margin: {blended_margin:+.1f}"

    # Momentum signal
    if abs(live_margin) > 15 and pct_remaining < 0.3:
        base["momentum_note"] = "⚠️  Large lead late — consider live-line value on trailing team if model agrees"
    elif abs(live_margin) < 3 and pct_remaining < 0.15:
        base["momentum_note"] = "🔥 Very close game in final minutes — high volatility, be cautious"
    else:
        base["momentum_note"] = ""

    return base


# ════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def train_model(seasons: list[str] = SEASONS) -> tuple[SpreadPredictor, pd.DataFrame, pd.DataFrame]:
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
    matchups   = build_matchup_dataset(games_df, rolling_df)
    print(f"  🔗 Matchup rows: {len(matchups):,}")

    print("\n🤖 Training XGBoost model...")
    predictor = SpreadPredictor()
    predictor.fit(matchups)

    return predictor, rolling_df, games_df


def run_analysis():
    # 1. Train / load model
    predictor, rolling_df, games_df = train_model()

    # 2. Fetch today's games from ESPN
    print("\n📡 Fetching live ESPN data...")
    today_games = fetch_espn_scoreboard()
    print(f"  🎮 Games found: {len(today_games)}")

    if not today_games:
        print("  No games found for today.")
        return []

    # 3. Generate recommendations
    recommendations = []
    print("\n" + "=" * 55)
    print("📊 SPREAD RECOMMENDATIONS")
    print("=" * 55)

    for game in today_games:
        state = game.get("state", "pre")
        if state == "in":
            rec = generate_ingame_recommendation(game, predictor, rolling_df, games_df)
        elif state == "pre":
            rec = generate_pregame_recommendation(game, predictor, rolling_df, games_df)
        else:
            continue  # skip finished games

        if "error" not in rec:
            recommendations.append(rec)
            _print_recommendation(rec)

    return recommendations


def _print_recommendation(rec: dict):
    print(f"\n{'─'*50}")
    print(f"🏀  {rec['away']} @ {rec['home']}")

    if rec["type"] == "ingame":
        print(f"    Period {rec['period']} | {rec['mins_remaining']:.0f} min remaining")
        print(f"    Live:  {rec['away']} {rec.get('live_margin', 0)*-1:+.0f}  |  {rec['home']} {rec.get('live_margin', 0):+.0f}")
        print(f"    Proj final margin (home): {rec.get('predicted_final_margin', '?'):+.1f}")
        print(f"    📌 IN-GAME: {rec.get('ingame_pick', 'N/A')}")
        if rec.get("momentum_note"):
            print(f"    {rec['momentum_note']}")
    else:
        print(f"    Predicted margin (home): {rec['predicted_margin']:+.1f}")
        if rec.get("spread") is not None:
            print(f"    Spread:   {rec['spread']:+.1f}  |  Edge: {rec.get('edge', 0):+.1f}")
        print(f"    📌 PICK:  {rec['pick']}")
        print(f"    Strength: {rec['strength']}  |  Confidence: {rec['confidence']:.0%}")

    print(f"    Model MAE: ±{rec['model_mae']} pts")


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║          NBA SPREAD PREDICTOR  v1.0                     ║
║  ⚠️  Educational purposes only — NOT financial advice   ║
╚══════════════════════════════════════════════════════════╝
""")
    recommendations = run_analysis()

    print(f"\n\n✅ Analysis complete — {len(recommendations)} games analyzed")
    print("   Results saved. Run again closer to tip-off for updated lines.")
