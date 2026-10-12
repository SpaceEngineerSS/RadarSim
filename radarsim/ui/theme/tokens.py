"""
Design tokens: the single source for colour, type, spacing and motion.

The look is an operator console: graphite surfaces separated by thin rules,
one amber accent for selection and warnings, and colour in the scopes used
only to carry meaning (track affiliation, detections, threshold lines).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    # Surfaces, darkest to lightest
    background: str
    surface: str
    raised: str
    hover: str
    line: str
    line_strong: str
    # Text
    text: str
    text_muted: str
    text_faint: str
    # Accent (selection, focus, warnings)
    accent: str
    accent_hover: str
    accent_text: str
    # States
    ok: str
    error: str
    # Scope
    scope_background: str
    scope_grid: str
    scope_grid_major: str
    scope_label: str
    sweep: str
    echo: str
    clutter: str
    threshold: str
    # Affiliation (NATO APP-6 hues, desaturated for long viewing)
    friend: str
    hostile: str
    neutral: str
    unknown: str
    expendable: str


DARK = Palette(
    background="#111315",
    surface="#171a1d",
    raised="#1e2226",
    hover="#262b30",
    line="#2a2f35",
    line_strong="#3a4148",
    text="#d5d9dd",
    text_muted="#8a929a",
    text_faint="#5b636b",
    accent="#e0a23a",
    accent_hover="#ebb459",
    accent_text="#16130c",
    ok="#74b98a",
    error="#e2685e",
    scope_background="#0b0d0e",
    scope_grid="#1c2226",
    scope_grid_major="#29323a",
    scope_label="#66717a",
    sweep="#9fc9c2",
    echo="#b9e0d9",
    clutter="#54646b",
    threshold="#e0a23a",
    friend="#5aa3cf",
    hostile="#dc6a5f",
    neutral="#7cb87f",
    unknown="#d6cc72",
    expendable="#9a8fb8",
)

LIGHT = Palette(
    background="#eef0f2",
    surface="#f7f8f9",
    raised="#ffffff",
    hover="#e6e9ec",
    line="#d6dadf",
    line_strong="#bcc2c9",
    text="#1c2126",
    text_muted="#59616a",
    text_faint="#8b939b",
    accent="#b97812",
    accent_hover="#a66a0c",
    accent_text="#ffffff",
    ok="#2f8a4c",
    error="#c2443a",
    scope_background="#0b0d0e",
    scope_grid="#1c2226",
    scope_grid_major="#29323a",
    scope_label="#66717a",
    sweep="#9fc9c2",
    echo="#b9e0d9",
    clutter="#54646b",
    threshold="#e0a23a",
    friend="#2f7fb0",
    hostile="#c2443a",
    neutral="#3f8f45",
    unknown="#a8952a",
    expendable="#6f62a0",
)


@dataclass(frozen=True)
class Typography:
    sans: str = "IBM Plex Sans"
    mono: str = "IBM Plex Mono"
    size_small: int = 11  # px
    size_body: int = 12
    size_title: int = 13
    size_display: int = 20


@dataclass(frozen=True)
class Spacing:
    xs: int = 4
    s: int = 8
    m: int = 12
    lg: int = 16
    xl: int = 24
    radius: int = 3
    rule: int = 1


@dataclass(frozen=True)
class Motion:
    """Durations in milliseconds; all motion uses an ease-out curve."""

    fast: int = 120
    normal: int = 180
    slow: int = 260


TYPE = Typography()
SPACE = Spacing()
MOTION = Motion()

AFFILIATIONS = ("friend", "hostile", "neutral", "unknown")


def affiliation_colour(palette: Palette, affiliation: str, target_type: str = "") -> str:
    if target_type in ("chaff", "decoy"):
        return palette.expendable
    return getattr(palette, affiliation, palette.unknown)
