"""
Шар L2: ознаки з постів каналу Повітряних сил (@kpszsu, веб-прев'ю; src/telegram_web.py).

Кожен пост → прапорці:
  тип загрози: drone (🛵 / БпЛА), jet (🏍 / реактивн), ballistic, missile (ракета, швидкісна
  ціль, крилат), kab (КАБ), shelter («в укриття», «терміново»),
  aviation (Ту-95/160/22, МіГ-31, стратегічна авіація, пуски, зліт);
  згадки областей: основи назв областей і великих міст (регулярні вирази нижче).

Ознаки в origin t (тільки пости з date_utc < t − BUFFER; пости з позначкою «edited» відкинуто):
  глобальні:   tg_n_{30m,1h,3h}, tg_{drone,jet,ballistic,missile,kab,shelter}_1h, tg_aviation_6h,
               tg_log_min_since_post
  по області:  tg_self_{30m,1h,3h} — згадки самої області (і її міст),
               tg_nb_{1h,3h}       — згадки сусідів «вгору за маршрутом»,
               tg_log_min_since_self
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

BUFFER = pd.Timedelta(minutes=5)
_NS_MIN = 60_000_000_000

TYPE_PATTERNS = {
    "drone": r"🛵|бпла|шахед|дрон",
    "jet": r"🏍|реактивн",
    "ballistic": r"балістик|балістичн",
    "missile": r"🚀|ракет|крилат|швидкісн\w*\s+ціл",
    "kab": r"💣|\bкаб",
    "aviation": r"ту-?95|ту-?160|ту-?22|міг-?31|стратегічн\w*\s+авіац|\bпуск|\bзліт",
    "shelter": r"укритт|терміново",
}

# Основи назв (нижній регістр). Не ідеально (річка Дніпро ≠ місто), але для лічильників досить.
REGION_PATTERNS = {
    "Kyiv City": r"\bки[єї]в(?!щ|ськ)|\bстолиц|видубич|троєщин|оболон|дарниц|святошин|позняк|осокорк|голосі"
                 r"|жулян|лук.янівк|нивк|шулявк|борщагів|теремк|печерськ|солом.янськ|поділ|подол|березняк"
                 r"|русанівк|виноградар|сирц|куренівк|харківськ\w* масив",
    "Kyivska oblast": r"київщин|київськ\w* обл|біл\w* церкв|бровар|борисп|васильк|обух|українк|фастів|фастов"
                      r"|ірпін|\bбуч[аіу]|вишгор|іванк|глевах|макарів|бородянк|березан|яготин|переяслав"
                      r"|кагарлик|миронівк|богуслав|тетіїв|сквир|чорнобил|славутич"
                      r"|боярк|вишнев|\bкозин|коцюбинськ|ворзел|гостомел|ірпін|бучанськ|білогородк|гатн",
    "Cherkaska oblast": r"черкас|\bуман|\bсміл|золотонош|канів|звенигород|корсун|\bшпол|жашк|христинівк"
                        r"|монастирищ|тальн|чигирин",
    "Kirovohradska oblast": r"кіровоград|кропивницьк|олександрі|світловодськ|знам.янк|новоукраїнк|гайворон"
                            r"|бобринец|долинськ|голованівськ|новомиргород|мал\w* виск|помічн",
    "Chernihivska oblast": r"чернігів|\bніжин|прилук|\bостер|козелец|бахмач|новгород-сіверськ|сновськ|корюківк|борзн",
    "Sumska oblast": r"\bсум(и|ах|ам|щин|ськ)|\bсум\b|шостк|конотоп|\bромн|охтирк|глухів|лебедин|тростянец|кролевец|путивл",
    "Poltavska oblast": r"полтав|кременчу|миргород|\bлубн|гадяч|кобеляк|горішн\w* плавн|пирятин|\bхорол|карлівк"
                        r"|решетилівк|зіньків",
    "Zhytomyrska oblast": r"житомир|коростен|бердичів|звягел|новоград|\bмалин|овруч|радомишл|коростишів|олевськ",
    "Dnipropetrovska oblast": r"дніпропетров|\bдніпр[оаіу]\b|кри\w+ р[іо]г|кривбас|кам.янськ|нікопол|павлоград"
                              r"|марганец|новомосковськ|синельников|жовт\w* вод",
    "Mykolaivska oblast": r"миколаї|вознесенськ|первомайськ|очаків|южноукраїнськ|баштанк|нов\w* буг",
    "Odeska oblast": r"\bодес|ізмаїл|чорноморськ|білгород-дністров|подільськ|затоков|татарбунар|березівк|\bрені\b|\bбалт[аі]\b",
}

GLOBAL_COLS = ["tg_n_30m", "tg_n_1h", "tg_n_3h", "tg_drone_1h", "tg_jet_1h", "tg_ballistic_1h",
               "tg_missile_1h", "tg_kab_1h", "tg_shelter_1h", "tg_aviation_6h", "tg_log_min_since_post"]
OBLAST_COLS = ["tg_self_30m", "tg_self_1h", "tg_self_3h", "tg_nb_1h", "tg_nb_3h", "tg_log_min_since_self"]
TG_COLS = GLOBAL_COLS + OBLAST_COLS


def prepare_posts(posts: pd.DataFrame) -> pd.DataFrame:
    """Відкидає відредаговані/порожні пости, додає прапорці типів і згадок областей."""
    p = posts[~posts["edited"] & (posts["text"].str.len() > 0)].copy()
    p["available_at"] = pd.to_datetime(p["date_utc"], utc=True)     # час публікації
    low = p["text"].str.lower().str.replace(r"\s+", " ", regex=True)   # nbsp і переноси → пробіл
    for k, pat in TYPE_PATTERNS.items():
        p[f"t_{k}"] = low.str.contains(pat, regex=True)
    for ob, pat in REGION_PATTERNS.items():
        p[f"r_{ob}"] = low.str.contains(pat.replace("(", "(?:").replace("(?:?", "(?"), regex=True)
    return p.sort_values("available_at").reset_index(drop=True)


def _counts(times_ns: np.ndarray, T_ns: np.ndarray, window_min: int) -> np.ndarray:
    """К-сть подій у [t − BUFFER − w, t − BUFFER)."""
    hi = T_ns - BUFFER.value
    lo = hi - window_min * _NS_MIN
    return (np.searchsorted(times_ns, hi, side="left") - np.searchsorted(times_ns, lo, side="left")).astype(float)


def _log_min_since(times_ns: np.ndarray, T_ns: np.ndarray, cap_min: int = 48 * 60) -> np.ndarray:
    hi = T_ns - BUFFER.value
    j = np.searchsorted(times_ns, hi, side="left") - 1
    since = np.where(j >= 0, (hi - times_ns[np.clip(j, 0, None)]) / _NS_MIN, cap_min)
    return np.log1p(np.clip(since, 0, cap_min))


def tg_features(posts: pd.DataFrame, origins: pd.DatetimeIndex, scope: list[str],
                neighbors: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    p = prepare_posts(posts)
    T = origins.as_unit("ns").asi8
    t_all = pd.DatetimeIndex(p["available_at"]).as_unit("ns").asi8
    sel = lambda m: t_all[m.to_numpy()]

    g = {"tg_n_30m": _counts(t_all, T, 30), "tg_n_1h": _counts(t_all, T, 60), "tg_n_3h": _counts(t_all, T, 180)}
    for k in ("drone", "jet", "ballistic", "missile", "kab", "shelter"):
        g[f"tg_{k}_1h"] = _counts(sel(p[f"t_{k}"]), T, 60)
    g["tg_aviation_6h"] = _counts(sel(p["t_aviation"]), T, 360)
    g["tg_log_min_since_post"] = _log_min_since(t_all, T)
    glob = pd.DataFrame(g, index=origins)

    per = {}
    for ob in scope:
        self_t = sel(p[f"r_{ob}"])
        nb_mask = np.zeros(len(p), dtype=bool)
        for nb in neighbors[ob]:
            if f"r_{nb}" in p:
                nb_mask |= p[f"r_{nb}"].to_numpy()
        nb_t = t_all[nb_mask]
        per[ob] = pd.DataFrame({
            "tg_self_30m": _counts(self_t, T, 30), "tg_self_1h": _counts(self_t, T, 60),
            "tg_self_3h": _counts(self_t, T, 180),
            "tg_nb_1h": _counts(nb_t, T, 60), "tg_nb_3h": _counts(nb_t, T, 180),
            "tg_log_min_since_self": _log_min_since(self_t, T),
        }, index=origins)
    return glob, per
