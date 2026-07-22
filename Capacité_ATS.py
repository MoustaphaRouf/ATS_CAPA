# -*- coding: utf-8 -*-
# ══════════════════════════════════════════════════════════════════════════
#  CAPACITÉ ATS — Estimation de la capacité d'un organe ATS par analyse
#  des communications vocales (Audio-DORATASK × Radio-Trafic)
#
#  Application Streamlit — Étape 2 du pipeline.
#
#  ┌────────────────────────────────────────────────────────────────────┐
#  │  ÉTAPE 1 (PRÉALABLE, HORS APPLICATION)                             │
#  │  L'isolation des prises de parole ATC versus Pilote et la          │
#  │  génération du fichier de charge de travail (classification        │
#  │  acoustique VAD + MFCC + Random Forest des enregistrements         │
#  │  air-sol) sont réalisées dans une première étape.                  │
#  │  → Contacter Roufaï : moustapharouf@yahoo.fr                       │
#  └────────────────────────────────────────────────────────────────────┘
#
#  ÉTAPE 2 (CETTE APPLICATION)
#  L'utilisateur charge les fichiers produits à l'étape 1 :
#    1. Charge audio  : CSV des segments classés (CTL / PILOTE)
#    2. Trafic        : statistiques de trafic (XLS/XLSX/CSV)
#  L'application génère :
#    • la courbe de charge radio (profil journalier moyen) ;
#    • les droites Audio-DORATASK et l'estimation de capacité DORATASK ;
#    • la courbe empirique Michaelis-Menten et la capacité empirique ;
#    • les intersections (φ_c*, C*) calibrant simultanément la fraction
#      vocale et la capacité, comparées à la capacité déclarée.
#
#  Méthodologie : article SSRN « Méthode d'estimation de la capacité d'un
#  organe ATS fondée sur l'analyse des communications vocales »
#  (Moustapha Amadou Roufaï, ASECNA Représentation du Bénin).
#  Implémentation dérivée du notebook CAPACITE_ATC_V11_5_I3_Michaelis.
#
#  Lancement :  streamlit run Capacité_ATS.py
# ══════════════════════════════════════════════════════════════════════════

import io
import re
import unicodedata
from datetime import datetime, time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, brentq

import warnings
warnings.filterwarnings("ignore")

# ── Style graphique homogène (identique au notebook / article) ────────────
matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200,
    "axes.titlelocation": "left"})
BLEU = "#1565C0"; ORANGE = "#EF6C00"; VERT = "#2E7D32"; ROUGE = "#E53935"
VIOLET = "#6A1B9A"; GRIS = "#888888"

LABELS_EST = {"mean": "Moyenne", "p50": "Médiane", "p70": "P70", "p85": "P85"}

__VERSION__ = "1.3 — dépôt ATS_CAPA câblé, listing Docs automatique"
COULEURS_EST = {"mean": GRIS, "p50": BLEU, "p70": ORANGE, "p85": ROUGE}
STYLES_EST = {"mean": ":", "p50": "-", "p70": "--", "p85": "-."}

PHI_C_RANGE = (0.02, 0.70)
PHI_C_VALUES = np.arange(PHI_C_RANGE[0], PHI_C_RANGE[1] + 0.0025, 0.005)

# ── Fichiers d'exemple hébergés sur GitHub (dossier Docs du dépôt) ────────
#  ⚠️ À METTRE À JOUR une fois le dépôt créé : remplacer <utilisateur>/<depot>
#  par les vôtres (ex. "roufai/capacite-ats"). Le contenu du dossier Docs est
#  listé automatiquement via l'API GitHub et proposé en listes déroulantes.
GITHUB_REPO = "MoustaphaRouf/ATS_CAPA"
GITHUB_BRANCHE = "main"
GITHUB_DOSSIER = "Docs"


# ══════════════════════════════════════════════════════════════════════════
#  1. FONCTIONS UTILITAIRES (pures — adaptées du notebook V11.4/V11.5)
# ══════════════════════════════════════════════════════════════════════════

_RE_HORO = re.compile(r"(\d{4})_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_ch\d+")


def parse_timestamp_fichier(chemin):
    """Timestamp de début d'enregistrement depuis le nom de fichier TopSky
    (format AAAA_M_J_H_M_S_chNN, ex. 2024_2_1_10_30_0_ch17)."""
    nom = Path(str(chemin)).name
    m = _RE_HORO.search(nom)
    if not m:
        return pd.NaT
    yr, mo, dy, hh, mm, ss = m.groups()
    return pd.Timestamp(f"{yr}-{mo.zfill(2)}-{dy.zfill(2)} "
                        f"{hh.zfill(2)}:{mm.zfill(2)}:{ss.zfill(2)}")


def reconstruire_horodatage(df_pred):
    """Ajoute t_debut / t_fin absolus au DataFrame de prédictions."""
    df = df_pred.copy()
    df["t_fichier"] = df["fichier_source"].apply(parse_timestamp_fichier)
    df["t_debut"] = df["t_fichier"] + pd.to_timedelta(df["debut_s"], unit="s")
    if "fin_s" in df.columns:
        df["t_fin"] = df["t_fichier"] + pd.to_timedelta(df["fin_s"], unit="s")
    else:
        df["t_fin"] = df["t_debut"] + pd.to_timedelta(
            df.get("duree_s", 0), unit="s")
    return df


def combiner_datetime(date, heure, decalage_h=0):
    """Combine une date et une heure Excel hétérogènes en Timestamp."""
    if pd.isna(date) or pd.isna(heure):
        return pd.NaT
    try:
        d = pd.Timestamp(date).date()
    except Exception:
        return pd.NaT
    ts = pd.NaT
    if isinstance(heure, dt_time):
        ts = pd.Timestamp.combine(d, heure)
    elif isinstance(heure, (pd.Timestamp, datetime)):
        ts = pd.Timestamp.combine(d, pd.Timestamp(heure).time())
    elif isinstance(heure, (int, float)) and not isinstance(heure, bool):
        frac = float(heure) % 1.0
        ts = pd.Timestamp(d) + pd.Timedelta(seconds=round(frac * 86400))
    else:
        s = str(heure).strip()
        if not s or s.lower() in ("nan", "nat", "none", "-"):
            return pd.NaT
        t = pd.to_datetime(s, errors="coerce")
        if pd.isna(t):
            return pd.NaT
        ts = pd.Timestamp.combine(d, t.time())
    if pd.notna(ts) and decalage_h:
        ts += pd.Timedelta(hours=decalage_h)
    return ts


def sans_accents(s):
    s = str(s)
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def norm_col(s):
    return re.sub(r"\s+", " ", sans_accents(str(s)).strip()).lower()


def trouver_colonne(df, nom, obligatoire=False):
    """Trouve une colonne par nom approché (accents / espaces tolérés)."""
    cible = norm_col(nom)
    mapping = {norm_col(c): c for c in df.columns}
    if cible in mapping:
        return mapping[cible]
    for k, v in mapping.items():
        if cible in k or k in cible:
            return v
    if obligatoire:
        raise KeyError(f"Colonne obligatoire absente : {nom}")
    return None


def lire_trafic_robuste(donnees, nom_fichier, col_date, col_entree, col_sortie):
    """Lit le fichier trafic (XLS/XLSX/CSV) en localisant l'en-tête."""
    suffix = Path(nom_fichier).suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(io.BytesIO(donnees))
        df.columns = [str(c).strip() for c in df.columns]
        return df.dropna(how="all")
    engine = "xlrd" if suffix == ".xls" else None
    raw = pd.read_excel(io.BytesIO(donnees), header=None, engine=engine)
    header_row = 0
    for i in range(min(30, len(raw))):
        vals = [norm_col(v) for v in raw.iloc[i].tolist()]
        if norm_col(col_date) in vals and (norm_col(col_entree) in vals
                                           or norm_col(col_sortie) in vals):
            header_row = i
            break
    df = pd.read_excel(io.BytesIO(donnees), header=header_row, engine=engine)
    df.columns = [str(c).strip() for c in df.columns]
    return df.dropna(how="all")


def charger_trafic(donnees, nom_fichier, date_debut, date_fin,
                   col_date, col_entree, col_sortie,
                   decalage_utc_h, transit_nominal_min):
    """Charge le trafic et construit t_entree / t_sortie."""
    df = lire_trafic_robuste(donnees, nom_fichier, col_date, col_entree,
                             col_sortie)
    c_date = trouver_colonne(df, col_date, obligatoire=True)
    c_entree = trouver_colonne(df, col_entree, obligatoire=True)
    c_sortie = trouver_colonne(df, col_sortie, obligatoire=True)
    df[c_date] = pd.to_datetime(df[c_date], errors="coerce")
    sem = df[(df[c_date] >= pd.Timestamp(date_debut)) &
             (df[c_date] <= pd.Timestamp(date_fin))].copy()
    sem["t_entree"] = sem.apply(lambda r: combiner_datetime(
        r[c_date], r[c_entree], decalage_utc_h), axis=1)
    sem["t_sortie"] = sem.apply(lambda r: combiner_datetime(
        r[c_date], r[c_sortie], decalage_utc_h), axis=1)
    m = sem["t_entree"].notna() & sem["t_sortie"].notna() \
        & (sem["t_sortie"] < sem["t_entree"])
    sem.loc[m, "t_sortie"] += pd.Timedelta(days=1)
    transit = pd.Timedelta(minutes=transit_nominal_min)
    m1 = sem["t_entree"].notna() & sem["t_sortie"].isna()
    sem.loc[m1, "t_sortie"] = sem.loc[m1, "t_entree"] + transit
    m2 = sem["t_sortie"].notna() & sem["t_entree"].isna()
    sem.loc[m2, "t_entree"] = sem.loc[m2, "t_sortie"] - transit
    return sem.dropna(subset=["t_entree", "t_sortie"]).copy()


# ── Modèles empiriques saturants ──────────────────────────────────────────

def michaelis_menten_origine(x, Cmax, k):
    """f(x) = Cmax · x / (k + x) — saturant, ramené à l'origine."""
    x = np.asarray(x, float)
    return Cmax * x / (k + x + 1e-12)


def ajuster_michaelis_menten(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 10:
        return None, None
    p0 = [max(y.max(), 1.0),
          max(np.median(x[x > 0]) if np.any(x > 0) else 0.05, 1e-3)]
    try:
        popt, _ = curve_fit(michaelis_menten_origine, x, y, p0=p0,
                            bounds=([0.1, 1e-6], [500.0, 10.0]), maxfev=20000)
        return popt, (lambda t: michaelis_menten_origine(
            np.asarray(t, float), *popt))
    except Exception:
        return None, None


def gompertz_origine(x, A, b, c):
    """f(x) = A · [exp(-exp(b - c·x)) - exp(-exp(b))] — Gompertz à l'origine."""
    return A * (np.exp(-np.exp(b - c * x)) - np.exp(-np.exp(b)))


def ajuster_gompertz(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 30:
        return None, None
    p0 = [max(y.max(), 1.0), 1.0, 8.0]
    try:
        popt, _ = curve_fit(gompertz_origine, x, y, p0=p0,
                            bounds=([0.1, -5.0, 0.1], [500.0, 10.0, 200.0]),
                            maxfev=20000)
        return popt, (lambda t: gompertz_origine(np.asarray(t, float), *popt))
    except Exception:
        return None, None


def metriques_reel(y_obs, y_pred):
    """R² et MAE sur les fenêtres réelles uniquement."""
    y_obs = np.asarray(y_obs, float); y_pred = np.asarray(y_pred, float)
    ok = np.isfinite(y_obs) & np.isfinite(y_pred)
    if ok.sum() < 2:
        return np.nan, np.nan
    ss_res = float(np.sum((y_obs[ok] - y_pred[ok]) ** 2))
    ss_tot = float(np.sum((y_obs[ok] - y_obs[ok].mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mae = float(np.mean(np.abs(y_obs[ok] - y_pred[ok])))
    return r2, mae


def intersection_continue(f_emp, pente, phi_min=0.03):
    """Première intersection non triviale entre C = pente·φ et f_emp(φ)."""
    grid = np.linspace(PHI_C_RANGE[0], PHI_C_RANGE[1], 900)

    def diff(p):
        return float(pente * p - f_emp(p))

    vals = np.array([diff(p) for p in grid], dtype=float)
    ok = np.isfinite(vals)
    grid, vals = grid[ok], vals[ok]
    if len(grid) < 3:
        return np.nan, np.nan
    for k in np.where(np.diff(np.sign(vals)) != 0)[0]:
        try:
            root = brentq(diff, grid[k], grid[k + 1])
            if root >= phi_min:
                return float(root), float(pente * root)
        except Exception:
            pass
    return np.nan, np.nan


def estimer_d_radio(ratios, mode):
    """Applique l'estimateur demandé à la série des ratios D_alpha/N."""
    r = pd.Series(ratios).replace([np.inf, -np.inf], np.nan).dropna()
    if mode == "mean":
        return float(r.mean())
    if mode == "p50":
        return float(r.quantile(0.50))
    if mode == "p70":
        return float(r.quantile(0.70))
    if mode == "p85":
        return float(r.quantile(0.85))
    raise ValueError(mode)


# ══════════════════════════════════════════════════════════════════════════
#  2. PIPELINE DE CALCUL (fonctions pures, mises en cache par l'UI)
# ══════════════════════════════════════════════════════════════════════════

def preparer_predictions(donnees_csv, seuil_confiance):
    """Filtre CTL/PILOTE au seuil de confiance et horodate les segments."""
    df_pred = pd.read_csv(io.BytesIO(donnees_csv))
    requis = {"fichier_source", "debut_s", "prediction", "confiance"}
    manquantes = requis - set(df_pred.columns)
    if manquantes:
        raise KeyError(
            "Colonnes absentes du fichier de charge : "
            + ", ".join(sorted(manquantes)))
    if "duree_s" not in df_pred.columns and "fin_s" not in df_pred.columns:
        raise KeyError("Le fichier de charge doit contenir duree_s ou fin_s.")
    if "duree_s" not in df_pred.columns:
        df_pred["duree_s"] = df_pred["fin_s"] - df_pred["debut_s"]
    df_pred = reconstruire_horodatage(df_pred)
    df_pred["prediction_norm"] = (df_pred["prediction"].astype(str)
                                  .str.strip().str.upper())
    df_radio = df_pred[
        (df_pred["prediction_norm"].isin(["CTL", "PILOTE"])) &
        (df_pred["confiance"] >= seuil_confiance)].copy()
    df_radio = df_radio.dropna(subset=["t_debut"])
    df_radio["heure"] = df_radio["t_debut"].dt.floor("h")
    return df_radio


def serie_horaire(df_radio, valides, alphas):
    """Série horaire : D_ctl, D_pil, D_alpha (par α) et N(h)."""
    charge_h = df_radio.groupby("heure").apply(lambda g: pd.Series({
        "D_ctl_s": g.loc[g["prediction_norm"] == "CTL", "duree_s"].sum(),
        "D_pil_s": g.loc[g["prediction_norm"] == "PILOTE", "duree_s"].sum(),
    })).reset_index()

    hmin = min(valides["t_entree"].min().floor("h"), charge_h["heure"].min())
    hmax = max(valides["t_sortie"].max().ceil("h"), charge_h["heure"].max())
    heures = pd.date_range(hmin, hmax, freq="h")

    ent = valides["t_entree"].values.astype("datetime64[ns]")
    bords = heures.values.astype("datetime64[ns]")
    N_h = np.histogram(ent.astype("int64"),
                       bins=np.append(bords.astype("int64"),
                                      (bords[-1] + np.timedelta64(1, "h"))
                                      .astype("int64")))[0]

    donnees_h = pd.DataFrame({"heure": heures, "N_h": N_h.astype(int)})
    donnees_h = donnees_h.merge(charge_h, on="heure", how="left")
    donnees_h[["D_ctl_s", "D_pil_s"]] = donnees_h[
        ["D_ctl_s", "D_pil_s"]].fillna(0.0)
    for a in alphas:
        donnees_h[f"D_alpha_{a}"] = (donnees_h["D_ctl_s"]
                                     + a * donnees_h["D_pil_s"])
    return donnees_h


def calculer_d_radio(donnees_h, alphas, estimateurs, percentile_pointe,
                     seuil_min_mvts, seuil_min_audio_s):
    """d_radio et pente DORATASK par (α, estimateur), heures de pointe."""
    a0 = alphas[0]
    actives = donnees_h[(donnees_h["N_h"] >= seuil_min_mvts) &
                        (donnees_h[f"D_alpha_{a0}"] >= seuil_min_audio_s)]
    if actives.empty:
        raise ValueError("Aucune heure active : vérifier la période et les "
                         "seuils (trafic minimal, durée audio minimale).")
    seuil_pointe = actives["N_h"].quantile(percentile_pointe)
    pointe = actives[actives["N_h"] >= seuil_pointe].copy()
    rows = []
    for a in alphas:
        ratios = pointe[f"D_alpha_{a}"] / pointe["N_h"]
        for est in estimateurs:
            d_r = estimer_d_radio(ratios, est)
            rows.append({"alpha": a, "estimateur": est,
                         "label": LABELS_EST[est],
                         "d_radio_s_mvt": round(d_r, 2),
                         "pente_dora": round(3600.0 / d_r, 2)})
    return pd.DataFrame(rows), len(actives), len(pointe), float(seuil_pointe)


def rasteriser(df_radio, valides):
    """Rasterisation à la seconde : [ctl, pilote, entrées] (vectorisée)."""
    t_min_r = df_radio["t_debut"].min().floor("min")
    t_max_r = df_radio["t_fin"].max().ceil("min")
    n_total = int((t_max_r - t_min_r).total_seconds()) + 1
    ts_arr = np.zeros((n_total, 3), dtype=np.float32)

    def secondes_depuis_t0(serie):
        """Index seconde depuis t_min_r, robuste aux unités datetime."""
        s = (pd.to_datetime(serie) - t_min_r).dt.total_seconds()
        return np.floor(s.to_numpy()).astype("int64")

    for col, cat in ((0, "CTL"), (1, "PILOTE")):
        sub = df_radio[df_radio["prediction_norm"] == cat]
        i0 = secondes_depuis_t0(sub["t_debut"])
        i1 = i0 + np.round(sub["duree_s"].to_numpy()).astype("int64")
        i0 = np.clip(i0, 0, n_total)
        i1 = np.clip(i1, 0, n_total)
        marque = np.zeros(n_total + 1, dtype=np.int32)
        np.add.at(marque, i0, 1)
        np.add.at(marque, i1, -1)
        ts_arr[:, col] = (np.cumsum(marque[:-1]) > 0).astype(np.float32)

    ie = secondes_depuis_t0(valides["t_entree"])
    ie = ie[(ie >= 0) & (ie < n_total)]
    np.add.at(ts_arr[:, 2], ie, 1.0)
    return ts_arr, n_total


def fenetres_reelles(ts_arr, n_total, win_min, alphas, step_min,
                     t_avant_min, t_apres_min):
    """X (par α) et Y2 pour toutes les fenêtres réelles de largeur win_min."""
    win_s = win_min * 60
    step_s = step_min * 60
    t_av, t_ap = t_avant_min * 60, t_apres_min * 60
    dur_y2_h = (win_s + t_av + t_ap) / 3600.0
    cs = np.vstack([np.zeros((1, 3)), np.cumsum(ts_arr, axis=0)])

    starts = np.arange(t_av, n_total - win_s - t_ap + 1, step_s)
    if len(starts) == 0:
        return pd.DataFrame()
    ends = starts + win_s
    d_ctl = cs[ends, 0] - cs[starts, 0]
    d_pil = cs[ends, 1] - cs[starts, 1]
    y2 = (cs[ends + t_ap, 2] - cs[starts - t_av, 2]) / dur_y2_h
    rec = {"win_min": win_min, "Y2": y2, "start_s": starts}
    for a in alphas:
        rec[f"X_a{a}"] = (d_ctl + a * d_pil) / win_s
    return pd.DataFrame(rec)


def superposition(ts_arr, n_total, win_min, alpha, step_min,
                  t_avant_min, t_apres_min, n_tirages, n_max, q_min, seed):
    """Scénarios synthétiques : superposition physique de fenêtres du top
    (1-q_min) de X, avec X et Y2 recalculés sur la série superposée."""
    rng = np.random.default_rng(seed)
    win_s = win_min * 60
    step_s = step_min * 60
    t_av, t_ap = t_avant_min * 60, t_apres_min * 60
    dur_y2_h = (win_s + t_av + t_ap) / 3600.0
    starts = list(range(t_av, n_total - win_s - t_ap + 1, step_s))
    if not starts:
        return pd.DataFrame()
    x_all = np.array([(ts_arr[s:s + win_s, 0].sum()
                       + alpha * ts_arr[s:s + win_s, 1].sum()) / win_s
                      for s in starts])
    seuil = np.quantile(x_all, q_min)
    top = [s for s, x in zip(starts, x_all) if x >= seuil]
    if len(top) < n_max + 1:
        return pd.DataFrame()
    recs = []
    for _ in range(n_tirages):
        idx = rng.choice(len(top), size=n_max, replace=False)
        acc = np.zeros((win_s, 3), dtype=np.float64)
        acc_y2 = np.zeros((win_s + t_av + t_ap, 3), dtype=np.float64)
        for lvl, i in enumerate(idx, start=1):
            s = top[i]
            acc += ts_arr[s:s + win_s]
            acc_y2 += ts_arr[s - t_av:s + win_s + t_ap]
            x = (acc[:, 0].sum() + alpha * acc[:, 1].sum()) / win_s
            if x <= 1.0:
                recs.append({"X": x, "Y2": acc_y2[:, 2].sum() / dur_y2_h,
                             "level": lvl})
    return pd.DataFrame(recs)


def telecharger_fichier(url, timeout=60):
    """Télécharge un fichier (URL raw GitHub) → (octets, nom_de_fichier)."""
    import urllib.request
    import urllib.parse
    req = urllib.request.Request(url, headers={"User-Agent": "CapaciteATS"})
    with urllib.request.urlopen(req, timeout=timeout) as rep:
        donnees = rep.read()
    nom = urllib.parse.unquote(url.rsplit("/", 1)[-1]) or "fichier"
    return donnees, nom


def lister_docs_github(depot, dossier=GITHUB_DOSSIER, branche=GITHUB_BRANCHE,
                       base_api="https://api.github.com", timeout=30):
    """Liste les fichiers du dossier `dossier` d'un dépôt GitHub public.

    Retourne une liste de dicts {nom, url, taille}, triée par nom.
    `depot` au format "utilisateur/depot".
    """
    import json
    import urllib.request
    url = (f"{base_api}/repos/{depot}/contents/{dossier}?ref={branche}")
    req = urllib.request.Request(
        url, headers={"User-Agent": "CapaciteATS",
                      "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as rep:
        contenu = json.loads(rep.read().decode("utf-8"))
    if isinstance(contenu, dict):   # message d'erreur API
        raise RuntimeError(contenu.get("message", "réponse API inattendue"))
    fichiers = [{"nom": f["name"], "url": f["download_url"],
                 "taille": f.get("size", 0)}
                for f in contenu
                if f.get("type") == "file" and f.get("download_url")]
    return sorted(fichiers, key=lambda f: f["nom"].lower())


# ══════════════════════════════════════════════════════════════════════════
#  3. INTERFACE STREAMLIT
# ══════════════════════════════════════════════════════════════════════════

def run_app():
    import streamlit as st

    st.set_page_config(page_title="Capacité ATS", page_icon="🗼",
                       layout="wide")

    st.title("🗼 Capacité ATS — Audio-DORATASK × Radio-Trafic")
    st.caption(
        "Estimation de la capacité d'un organe ATS par analyse des "
        "communications vocales — méthodologie de l'article SSRN "
        f"(M. A. Roufaï, ASECNA). — **Version {__VERSION__}**")

    st.info(
        "**Étape 1 (préalable, hors application)** — L'isolation des prises "
        "de parole **ATC versus Pilote** et la **génération du fichier de "
        "charge de travail** (classification acoustique VAD + MFCC + Random "
        "Forest des enregistrements air-sol) sont réalisées dans une "
        "première étape. 📧 Contacter **Roufaï** : "
        "**moustapharouf@yahoo.fr**\n\n"
        "**Étape 2 (cette application)** — Chargez ci-dessous les fichiers "
        "issus de l'étape 1 (charge audio, trafic) : l'application génère la "
        "courbe de charge, la courbe DORATASK et l'estimation de capacité "
        "DORATASK, la courbe empirique et la capacité empirique, ainsi que "
        "leur intersection (φ_c*, C*).", icon="ℹ️")

    # ── Barre latérale : paramètres ──────────────────────────────────────
    sb = st.sidebar
    sb.header("⚙️ Paramètres")

    sb.subheader("Période d'étude")
    date_debut = sb.date_input("Début", value=pd.Timestamp("2024-02-01"))
    date_fin = sb.date_input("Fin", value=pd.Timestamp("2024-02-28"))

    sb.subheader("Charge vocale")
    seuil_confiance = sb.slider("Seuil de confiance du classifieur",
                                0.50, 0.95, 0.70, 0.05)
    alphas = sb.multiselect(
        "α — poids de la parole pilote (paramètre de terrain)",
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        default=[0.2, 0.7])
    alphas = sorted(alphas) or [0.2, 0.7]

    sb.subheader("Audio-DORATASK")
    estimateurs = sb.multiselect(
        "Estimateurs de d_radio", ["mean", "p50", "p70", "p85"],
        default=["mean", "p50", "p70", "p85"],
        format_func=lambda e: LABELS_EST[e])
    estimateurs = estimateurs or ["p70", "p85"]
    percentile_pointe = sb.slider("Percentile heures de pointe",
                                  0.50, 0.95, 0.75, 0.05)
    seuil_min_mvts = sb.number_input("Trafic horaire minimal (mvts/h)",
                                     1, 20, 1)
    seuil_min_audio = sb.number_input("Durée audio horaire minimale (s)",
                                      0, 600, 30)

    sb.subheader("Modèle empirique (Radio-Trafic)")
    win_min = sb.selectbox("Fenêtre glissante W (min)",
                           [60, 90, 120, 180], index=3)
    modele_choix = sb.radio("Modèle saturant",
                            ["Michaelis-Menten",
                             "Michaelis-Menten + Gompertz (sensibilité)"])
    step_min = sb.number_input("Pas de glissement (min)", 1, 30, 5)
    t_avant = sb.number_input("Fenêtre trafic Y2 — extension avant (min)",
                              0, 30, 10)
    t_apres = sb.number_input("Fenêtre trafic Y2 — extension après (min)",
                              0, 30, 5)
    n_tirages = sb.number_input("Tirages de superposition synthétique",
                                500, 10000, 3000, step=500)
    n_max = sb.number_input("Niveaux de superposition (1→N)", 2, 6, 4)
    q_min = sb.slider("Quantile bas des tirages (top X)", 0.50, 0.90,
                      0.70, 0.05)
    seed = sb.number_input("Graine aléatoire", 0, 9999, 42)

    sb.subheader("Trafic")
    col_date = sb.text_input("Colonne date", "Date du vol")
    col_entree = sb.text_input("Colonne heure d'entrée", "Hreentrée")
    col_sortie = sb.text_input("Colonne heure de sortie", "Hresortie")
    decalage_utc = sb.number_input("Décalage trafic → UTC (h)", -12, 12, 0)
    transit_nominal = sb.number_input("Transit nominal TMA (min)", 1, 60, 12)

    sb.subheader("Référence")
    capa_declaree = sb.number_input("Capacité déclarée (vols/h)",
                                    1, 120, 21)

    # ── Chargement des fichiers ──────────────────────────────────────────
    st.header("1️⃣ Données d'entrée (issues de l'étape 1)")

    source = st.radio(
        "Source des données",
        ["📤 Téléverser mes fichiers",
         "🧪 Tester avec les fichiers d'exemple (GitHub / Docs)"],
        horizontal=True)

    octets_pred = octets_traf = None
    nom_traf = ""

    if source.startswith("📤"):
        c1, c2 = st.columns(2)
        with c1:
            f_pred = st.file_uploader(
                "Charge audio — segments classés CTL / PILOTE (CSV)",
                type=["csv"],
                help="Colonnes attendues : fichier_source, debut_s, duree_s "
                     "(ou fin_s), prediction, confiance. Produit à "
                     "l'étape 1 — contacter moustapharouf@yahoo.fr.")
        with c2:
            f_traf = st.file_uploader(
                "Statistiques de trafic (XLS / XLSX / CSV)",
                type=["xls", "xlsx", "csv"],
                help="Doit contenir la date du vol et les heures d'entrée / "
                     "sortie de l'espace aérien étudié.")
        if not (f_pred and f_traf):
            st.warning("Chargez les deux fichiers pour lancer l'analyse — "
                       "ou basculez sur les fichiers d'exemple pour tester "
                       "l'application.")
            st.stop()
        octets_pred = f_pred.getvalue()
        octets_traf = f_traf.getvalue()
        nom_traf = f_traf.name
    else:
        st.markdown(
            "Les fichiers d'exemple sont chargés automatiquement depuis le "
            f"dossier **{GITHUB_DOSSIER}** du dépôt "
            f"[{GITHUB_REPO}](https://github.com/{GITHUB_REPO}) — "
            "choisissez simplement vos fichiers dans les listes "
            "déroulantes.")

        with st.expander("⚙️ Source avancée (autre dépôt / branche / "
                         "dossier)", expanded=False):
            depot = st.text_input("Dépôt GitHub (utilisateur/depot)",
                                  GITHUB_REPO)
            branche = st.text_input("Branche", GITHUB_BRANCHE)
            dossier = st.text_input("Dossier", GITHUB_DOSSIER)

        @st.cache_data(ttl=600,
                       show_spinner="Lecture du dossier Docs sur GitHub…")
        def _lister(d, doss, br):
            return lister_docs_github(d, doss, br)

        try:
            fichiers = _lister(depot.strip(), dossier.strip(),
                               branche.strip())
        except Exception as e:
            st.error(
                f"Impossible de lister le dossier {dossier} du dépôt "
                f"{depot} : {e}. Causes possibles : pas de connexion "
                "Internet, limite de débit de l'API GitHub atteinte "
                "(réessayer dans quelques minutes), ou dossier absent de "
                "cette branche. En attendant, la source « Téléverser mes "
                "fichiers » reste disponible.")
            st.stop()

        def _ko(t):
            return f"{t/1e6:.1f} Mo" if t >= 1e6 else f"{t/1e3:.0f} ko"

        # Les deux listes déroulantes montrent TOUT le contenu du dossier
        # Docs ; l'utilisateur choisit lui-même le fichier de prédictions
        # et le fichier de statistiques.
        tous_noms = [f["nom"] for f in fichiers]
        if not tous_noms:
            st.error(f"Le dossier {dossier} du dépôt {depot} est vide.")
            st.stop()

        etiquettes = {f["nom"]: f"{f['nom']}  ({_ko(f['taille'])})"
                      for f in fichiers}
        urls = {f["nom"]: f["url"] for f in fichiers}

        # Présélections raisonnables dans le contenu du dossier :
        # "predi"/"charge" → prédictions ; "statistique"/"trafic"/"survol"
        # → statistiques.
        def _defaut(noms, mots):
            for i, n in enumerate(noms):
                if any(m in n.lower() for m in mots):
                    return i
            return 0

        s1, s2 = st.columns(2)
        nom_charge = s1.selectbox(
            "📄 Prédictions — segments classés CTL/PILOTE",
            tous_noms,
            index=_defaut(tous_noms, ["predi", "charge"]),
            format_func=lambda n: etiquettes[n],
            help="Contenu du dossier Docs — choisir le CSV de prédictions "
                 "acoustiques produit à l'étape 1.")
        nom_traf_sel = s2.selectbox(
            "📊 Statistiques de trafic",
            tous_noms,
            index=_defaut(tous_noms, ["statistique", "trafic", "survol"]),
            format_func=lambda n: etiquettes[n],
            help="Contenu du dossier Docs — choisir le fichier XLS/XLSX/CSV "
                 "des statistiques de trafic (entrées/sorties).")

        if not nom_charge.lower().endswith(".csv"):
            st.error(f"« {nom_charge} » n'est pas un CSV : le fichier de "
                     "prédictions doit être au format CSV.")
            st.stop()
        if not nom_traf_sel.lower().endswith((".xls", ".xlsx", ".csv")):
            st.error(f"« {nom_traf_sel} » n'est pas un fichier de trafic "
                     "exploitable (formats acceptés : XLS, XLSX, CSV).")
            st.stop()
        if nom_charge == nom_traf_sel:
            st.warning("Le même fichier est sélectionné pour la charge et "
                       "le trafic — vérifiez votre choix.")

        @st.cache_data(show_spinner="Téléchargement des fichiers "
                                    "d'exemple depuis GitHub…")
        def _demo(u1, u2):
            p, _ = telecharger_fichier(u1)
            t, n = telecharger_fichier(u2)
            return p, t, n

        try:
            octets_pred, octets_traf, nom_traf = _demo(urls[nom_charge],
                                                       urls[nom_traf_sel])
        except Exception as e:
            st.error(f"Téléchargement impossible depuis GitHub : {e}.")
            st.stop()
        st.success(f"Fichiers d'exemple chargés : « {nom_charge} » "
                   f"({len(octets_pred)/1e6:.1f} Mo) et « {nom_traf} » "
                   f"({_ko(len(octets_traf))}).")

    # ── Pipeline (mis en cache sur le contenu des fichiers + paramètres) ─
    @st.cache_data(show_spinner="Lecture des prédictions acoustiques…")
    def _etape_pred(donnees, seuil):
        return preparer_predictions(donnees, seuil)

    @st.cache_data(show_spinner="Lecture du fichier trafic…")
    def _etape_trafic(donnees, nom, d0, d1, cd, ce, cs, dec, tn):
        return charger_trafic(donnees, nom, d0, d1, cd, ce, cs, dec, tn)

    @st.cache_data(show_spinner="Rasterisation à la seconde…")
    def _etape_raster(donnees_pred, seuil, donnees_traf, nom, d0, d1,
                      cd, ce, cs, dec, tn):
        dr = preparer_predictions(donnees_pred, seuil)
        va = charger_trafic(donnees_traf, nom, d0, d1, cd, ce, cs, dec, tn)
        return rasteriser(dr, va)

    try:
        df_radio = _etape_pred(octets_pred, seuil_confiance)
        valides = _etape_trafic(octets_traf, nom_traf,
                                str(date_debut), str(date_fin),
                                col_date, col_entree, col_sortie,
                                int(decalage_utc), int(transit_nominal))
    except KeyError as e:
        st.error(f"Fichier invalide : {e}")
        st.stop()

    if df_radio.empty or valides.empty:
        st.error("Aucun segment vocal fiable ou aucun vol valide sur la "
                 "période — vérifier les dates et les fichiers.")
        st.stop()

    n_ctl = int((df_radio["prediction_norm"] == "CTL").sum())
    n_pil = int((df_radio["prediction_norm"] == "PILOTE").sum())
    donnees_h = serie_horaire(df_radio, valides, alphas)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Segments CTL fiables", f"{n_ctl:,}".replace(",", " "))
    m2.metric("Segments PILOTE fiables", f"{n_pil:,}".replace(",", " "))
    m3.metric("Vols valides", f"{len(valides):,}".replace(",", " "))
    m4.metric("Heures de la série", f"{len(donnees_h):,}".replace(",", " "))

    # ── 2. Courbe de charge ──────────────────────────────────────────────
    st.header("2️⃣ Courbe de charge radio — profil journalier moyen")
    prof = donnees_h.copy()
    prof["hh"] = prof["heure"].dt.hour
    agg = prof.groupby("hh").agg(
        ctl_moy=("D_ctl_s", "mean"),
        pil_moy=("D_pil_s", "mean"),
        ctl_p25=("D_ctl_s", lambda s: s.quantile(0.25)),
        ctl_p75=("D_ctl_s", lambda s: s.quantile(0.75))).reset_index()

    fig1, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.fill_between(agg["hh"], agg["ctl_p25"], agg["ctl_p75"],
                    color=BLEU, alpha=0.15, label="IQR CTL (P25–P75)")
    ax.plot(agg["hh"], agg["ctl_moy"], color=BLEU, marker="o", ms=3,
            lw=1.8, label="Contrôleur (CTL)")
    ax.plot(agg["hh"], agg["pil_moy"], color=ORANGE, marker="o", ms=3,
            lw=1.8, label="Pilote")
    pic = agg.loc[agg["ctl_moy"].idxmax()]
    ax.annotate(f"pic {pic['ctl_moy']:.0f} s/h à {int(pic['hh'])}h",
                xy=(pic["hh"], pic["ctl_moy"]),
                xytext=(pic["hh"] + 1.2, pic["ctl_moy"] * 1.02),
                fontsize=8, color=GRIS,
                arrowprops=dict(arrowstyle="-", color=GRIS, lw=0.7))
    ax.set_xlabel("Heure de la journée")
    ax.set_ylabel("Temps de parole (s / heure)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title("Profil journalier moyen de la charge radio")
    ax.legend()
    st.pyplot(fig1)

    occ_pic = (pic["ctl_moy"] + pic["pil_moy"]) / 3600 * 100
    st.caption(
        f"Au pic ({int(pic['hh'])}h), la charge cumulée CTL+Pilote "
        f"représente ≈ {occ_pic:.1f} % de la durée disponible.")

    # ── 3. Audio-DORATASK ────────────────────────────────────────────────
    st.header("3️⃣ Audio-DORATASK — C(φ_c) = φ_c × 3600 / d_radio")
    try:
        d_radio_df, n_act, n_pte, seuil_pte = calculer_d_radio(
            donnees_h, alphas, estimateurs, percentile_pointe,
            int(seuil_min_mvts), int(seuil_min_audio))
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.caption(f"Heures actives : {n_act} — heures de pointe "
               f"(N ≥ {seuil_pte:.0f}) : {n_pte}")
    st.dataframe(
        d_radio_df.rename(columns={
            "alpha": "α", "label": "Estimateur",
            "d_radio_s_mvt": "d_radio (s/mvt)",
            "pente_dora": "Pente DORATASK"})[
            ["α", "Estimateur", "d_radio (s/mvt)", "Pente DORATASK"]],
        use_container_width=True, hide_index=True)

    phi_illustratif = st.slider(
        "φ_c illustratif — fraction vocale supposée de la charge de travail "
        "(lecture directe de la capacité DORATASK)",
        float(PHI_C_RANGE[0]), float(PHI_C_RANGE[1]), 0.30, 0.01)

    fig2, axes2 = plt.subplots(1, len(alphas),
                               figsize=(6.2 * len(alphas), 4.6), sharey=True)
    if len(alphas) == 1:
        axes2 = [axes2]
    for ax, a in zip(axes2, alphas):
        sub = d_radio_df[d_radio_df["alpha"] == a]
        for _, r in sub.iterrows():
            C = PHI_C_VALUES * 3600.0 / r["d_radio_s_mvt"]
            ax.plot(PHI_C_VALUES, C, color=COULEURS_EST[r["estimateur"]],
                    ls=STYLES_EST[r["estimateur"]], lw=2,
                    label=f"{r['label']} (d={r['d_radio_s_mvt']:.0f} s/mvt)")
        ax.axvline(phi_illustratif, color=VERT, ls="--", lw=1.4)
        ax.axhline(capa_declaree, color=GRIS, ls="-", lw=1, alpha=0.6)
        ax.text(PHI_C_RANGE[1] * 0.99, capa_declaree + 0.6,
                f"Déclarée {capa_declaree}/h", color=GRIS, fontsize=7.5,
                ha="right")
        ax.set_xlabel("φ_c — fraction vocale de la charge de travail")
        ax.set_title(f"α = {a}")
        ax.legend(loc="upper left", title="Estimateur de d_radio")
    axes2[0].set_ylabel("C_DORA (vols/h)")
    st.pyplot(fig2)

    lignes = []
    for a in alphas:
        sub = d_radio_df[d_radio_df["alpha"] == a]
        for _, r in sub.iterrows():
            lignes.append({
                "α": a, "Estimateur": r["label"],
                f"C_DORA à φ_c={phi_illustratif:.2f} (vols/h)":
                    round(phi_illustratif * 3600 / r["d_radio_s_mvt"], 1)})
    st.markdown(f"**Estimation de capacité DORATASK à φ_c = "
                f"{phi_illustratif:.2f}** *(lecture illustrative — la "
                f"détermination de φ_c est faite par l'intersection, "
                f"section 5)* :")
    st.dataframe(pd.DataFrame(lignes), use_container_width=True,
                 hide_index=True)

    # ── 4. Courbe empirique ──────────────────────────────────────────────
    st.header("4️⃣ Courbe empirique Radio-Trafic (modèle saturant)")

    @st.cache_data(show_spinner="Fenêtres glissantes et superposition "
                                "synthétique — quelques minutes possibles…")
    def _etape_empirique(donnees_pred, seuil, donnees_traf, nom, d0, d1,
                         cd, ce, cs, dec, tn, wm, alphas_t, stp, tav, tap,
                         ntir, nmax, qmin, sd):
        ts_arr, n_total = _etape_raster(donnees_pred, seuil, donnees_traf,
                                        nom, d0, d1, cd, ce, cs, dec, tn)
        df_r = fenetres_reelles(ts_arr, n_total, wm, list(alphas_t),
                                stp, tav, tap)
        synths = {a: superposition(ts_arr, n_total, wm, a, stp, tav, tap,
                                   ntir, nmax, qmin, sd)
                  for a in alphas_t}
        return df_r, synths

    df_reel, synths = _etape_empirique(
        octets_pred, seuil_confiance, octets_traf, nom_traf,
        str(date_debut), str(date_fin), col_date, col_entree, col_sortie,
        int(decalage_utc), int(transit_nominal), int(win_min),
        tuple(alphas), int(step_min), int(t_avant), int(t_apres),
        int(n_tirages), int(n_max), float(q_min), int(seed))

    if df_reel.empty:
        st.error("Aucune fenêtre glissante constructible — période trop "
                 "courte pour la fenêtre W choisie.")
        st.stop()

    modeles = {}
    metriques = []
    for a in alphas:
        df_r = (df_reel[[f"X_a{a}", "Y2"]]
                .rename(columns={f"X_a{a}": "X"}).copy())
        df_r = df_r.replace([np.inf, -np.inf], np.nan).dropna()
        df_r = df_r[df_r["Y2"] > 0]
        df_s = synths.get(a, pd.DataFrame())
        if len(df_s):
            df_fit = pd.concat([df_r.assign(level=0),
                                df_s[["X", "Y2", "level"]]],
                               ignore_index=True)
        else:
            df_fit = df_r.assign(level=0)

        popt, f_mm = ajuster_michaelis_menten(df_fit["X"], df_fit["Y2"])
        if f_mm is not None:
            r2, mae = metriques_reel(df_r["Y2"].values,
                                     f_mm(df_r["X"].values))
            modeles[(a, "Michaelis-Menten")] = f_mm
            metriques.append({
                "α": a, "Modèle": "Michaelis-Menten",
                "R² (réel)": round(r2, 3), "MAE (réel)": round(mae, 2),
                "Fenêtres réelles": len(df_r),
                "Points synthétiques": len(df_s),
                "Paramètres": f"Cmax={popt[0]:.1f}, k={popt[1]:.3f}"})
        if modele_choix.endswith("(sensibilité)"):
            poptg, f_go = ajuster_gompertz(df_fit["X"], df_fit["Y2"])
            if f_go is not None:
                r2g, maeg = metriques_reel(df_r["Y2"].values,
                                           f_go(df_r["X"].values))
                modeles[(a, "Gompertz-origine")] = f_go
                metriques.append({
                    "α": a, "Modèle": "Gompertz-origine",
                    "R² (réel)": round(r2g, 3),
                    "MAE (réel)": round(maeg, 2),
                    "Fenêtres réelles": len(df_r),
                    "Points synthétiques": len(df_s),
                    "Paramètres": np.round(poptg, 3).tolist()})

    if not metriques:
        st.error("Échec de l'ajustement du modèle empirique — données "
                 "insuffisantes.")
        st.stop()

    st.dataframe(pd.DataFrame(metriques), use_container_width=True,
                 hide_index=True)
    st.caption("Les métriques R² et MAE sont calculées sur les fenêtres "
               "réelles uniquement ; l'ajustement utilise les fenêtres "
               "réelles augmentées des scénarios synthétiques de "
               "superposition.")

    # ── 5. Intersections ─────────────────────────────────────────────────
    st.header("5️⃣ Intersections Audio-DORATASK × courbe empirique "
              "(φ_c*, C*)")

    inter_rows = []
    fig3, axes3 = plt.subplots(1, len(alphas),
                               figsize=(6.8 * len(alphas), 5.2), sharey=True)
    if len(alphas) == 1:
        axes3 = [axes3]

    for ax, a in zip(axes3, alphas):
        f_mm = modeles.get((a, "Michaelis-Menten"))
        df_r = (df_reel[[f"X_a{a}", "Y2"]]
                .rename(columns={f"X_a{a}": "X"}))
        ax.scatter(df_r["X"], df_r["Y2"], s=8, alpha=0.12, color=GRIS,
                   label="Fenêtres réelles")

        if f_mm is not None:
            met = [m for m in metriques if m["α"] == a
                   and m["Modèle"] == "Michaelis-Menten"][0]
            ax.plot(PHI_C_VALUES, f_mm(PHI_C_VALUES), color=VERT, lw=3.0,
                    label=(f"Michaelis-Menten (W={win_min} min, "
                           f"R²={met['R² (réel)']:.2f}, "
                           f"MAE={met['MAE (réel)']:.2f})"))
        f_go = modeles.get((a, "Gompertz-origine"))
        if f_go is not None:
            ax.plot(PHI_C_VALUES, f_go(PHI_C_VALUES), color=VIOLET,
                    lw=1.8, ls="--", label="Gompertz-origine (sensibilité)")

        for est in estimateurs:
            r = d_radio_df[(d_radio_df["alpha"] == a) &
                           (d_radio_df["estimateur"] == est)].iloc[0]
            pente = 3600.0 / r["d_radio_s_mvt"]
            ax.plot(PHI_C_VALUES, pente * PHI_C_VALUES,
                    color=COULEURS_EST[est], ls=STYLES_EST[est], lw=1.8,
                    label=f"Audio-DORATASK {r['label']}")
            for nom_mod, f_emp in ((("Michaelis-Menten"), f_mm),
                                   (("Gompertz-origine"), f_go)):
                if f_emp is None:
                    continue
                phi_i, C_i = intersection_continue(f_emp, pente)
                ok = np.isfinite(phi_i) and np.isfinite(C_i)
                inter_rows.append({
                    "α": a, "Modèle": nom_mod, "Estimateur": r["label"],
                    "d_radio (s/mvt)": r["d_radio_s_mvt"],
                    "Intersection": "oui" if ok else "—",
                    "φ_c*": round(phi_i, 3) if ok else np.nan,
                    "C* (vols/h)": round(C_i, 1) if ok else np.nan,
                    "Écart vs déclarée (%)":
                        round((C_i - capa_declaree) / capa_declaree * 100, 1)
                        if ok else np.nan})
                if ok and nom_mod == "Michaelis-Menten":
                    ax.scatter([phi_i], [C_i], color=COULEURS_EST[est],
                               zorder=7, s=90, marker="*",
                               edgecolor="white", linewidth=0.6)
                    ax.annotate(f"{r['label']}\n{C_i:.1f} v/h",
                                xy=(phi_i, C_i), xytext=(5, 5),
                                textcoords="offset points", fontsize=7,
                                color=COULEURS_EST[est])

        ax.axhline(capa_declaree, color=GRIS, lw=1.0, alpha=0.7)
        ax.text(PHI_C_RANGE[1] * 0.99, capa_declaree + 0.6,
                f"Déclarée {capa_declaree}/h", color=GRIS, fontsize=7.5,
                ha="right")
        ax.set_xlabel("φ_c — fraction vocale de la charge de travail")
        ax.set_title(f"α = {a} | W = {win_min} min")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=7)
    axes3[0].set_ylabel("Capacité / débit estimé (vols/h)")
    st.pyplot(fig3)

    df_inter = pd.DataFrame(inter_rows)
    st.dataframe(df_inter, use_container_width=True, hide_index=True)

    mm_ok = df_inter[(df_inter["Modèle"] == "Michaelis-Menten") &
                     (df_inter["Intersection"] == "oui")]
    if len(mm_ok):
        p70 = mm_ok[mm_ok["Estimateur"] == "P70"]
        if len(p70):
            meilleur = p70.sort_values("α", ascending=False).iloc[0]
            st.success(
                f"**Scénario nominal prudent (P70, α={meilleur['α']})** : "
                f"φ_c* ≈ {meilleur['φ_c*']:.2f} → "
                f"**C* ≈ {meilleur['C* (vols/h)']:.1f} vols/h** "
                f"(capacité déclarée : {capa_declaree} vols/h, écart "
                f"{meilleur['Écart vs déclarée (%)']:+.1f} %). Les valeurs "
                f"P85 sont des bornes hautes de sensibilité, non des "
                f"capacités directement déclarables.")
        cmin = mm_ok["C* (vols/h)"].min()
        cmax = mm_ok["C* (vols/h)"].max()
        st.caption(f"Plage de calibration Michaelis-Menten : "
                   f"C* ∈ [{cmin:.1f} ; {cmax:.1f}] vols/h.")
    else:
        st.warning(
            "Aucune intersection trouvée : la pente empirique initiale est "
            "inférieure aux pentes DORATASK. Vérifier d_radio, la fenêtre W "
            "et le domaine de X (augmenter les tirages synthétiques peut "
            "étendre le domaine).")

    # ── 6. Exports ───────────────────────────────────────────────────────
    st.header("6️⃣ Exports")
    e1, e2, e3 = st.columns(3)
    e1.download_button(
        "⬇️ Série horaire (CSV)",
        donnees_h.to_csv(index=False).encode("utf-8-sig"),
        "serie_horaire.csv", "text/csv")
    e2.download_button(
        "⬇️ Estimateurs d_radio (CSV)",
        d_radio_df.to_csv(index=False).encode("utf-8-sig"),
        "d_radio_estimateurs.csv", "text/csv")
    e3.download_button(
        "⬇️ Intersections φ_c*, C* (CSV)",
        df_inter.to_csv(index=False).encode("utf-8-sig"),
        "intersections.csv", "text/csv")

    st.divider()
    st.caption(
        "Capacité ATS — pipeline Audio-DORATASK × Radio-Trafic. "
        "Étape 1 (isolation ATC/Pilote, génération du fichier de charge) : "
        "contacter Roufaï — moustapharouf@yahoo.fr. Les capacités issues "
        "des quantiles élevés (P85) sont des bornes de sensibilité "
        "nécessitant une validation opérationnelle complémentaire.")


if __name__ == "__main__":
    run_app()
