"""
Application stylesheet generated from the design tokens.

Widgets opt into variants through dynamic properties, e.g.
``button.setProperty("variant", "primary")`` or
``label.setProperty("role", "caption")``.
"""

from .tokens import SPACE, TYPE, Palette


def build_stylesheet(p: Palette) -> str:
    r = SPACE.radius
    return f"""
* {{
    font-family: "{TYPE.sans}";
    font-size: {TYPE.size_body}px;
    color: {p.text};
    outline: none;
}}
QMainWindow, QDialog, QWidget#Root {{ background: {p.background}; }}
QWidget {{ background: transparent; }}
QFrame#Panel, QWidget#Panel {{ background: {p.surface}; }}
QFrame[rule="right"] {{ border-right: 1px solid {p.line}; }}
QFrame[rule="left"] {{ border-left: 1px solid {p.line}; }}
QFrame[rule="top"] {{ border-top: 1px solid {p.line}; }}
QFrame[rule="bottom"] {{ border-bottom: 1px solid {p.line}; }}
QFrame#Separator {{ background: {p.line}; max-height: 1px; min-height: 1px; border: none; }}

QLabel[role="section"] {{
    color: {p.text_muted};
    font-size: {TYPE.size_small}px;
    font-weight: 600;
    letter-spacing: 0.6px;
    padding: {SPACE.s}px 0 {SPACE.xs}px 0;
}}
QLabel[role="caption"] {{ color: {p.text_muted}; font-size: {TYPE.size_small}px; }}
QLabel[role="faint"] {{ color: {p.text_faint}; font-size: {TYPE.size_small}px; }}
QLabel[role="value"] {{ font-family: "{TYPE.mono}"; }}
QLabel[role="title"] {{ font-size: {TYPE.size_title}px; font-weight: 600; }}
QLabel[role="display"] {{ font-family: "{TYPE.mono}"; font-size: {TYPE.size_display}px; }}
QLabel[role="badge"] {{
    font-family: "{TYPE.mono}"; font-size: {TYPE.size_small}px;
    border: 1px solid {p.line_strong}; border-radius: {r}px; padding: 1px 6px;
    color: {p.text_muted};
}}
QLabel[role="badge"][state="live"] {{ color: {p.ok}; border-color: {p.ok}; }}
QLabel[role="badge"][state="paused"] {{ color: {p.accent}; border-color: {p.accent}; }}
QLabel[role="badge"][state="replay"] {{ color: {p.friend}; border-color: {p.friend}; }}
QLabel[role="badge"][state="edit"] {{ color: {p.unknown}; border-color: {p.unknown}; }}

QPushButton {{
    background: {p.raised};
    border: 1px solid {p.line_strong};
    border-radius: {r}px;
    padding: 5px 12px;
}}
QPushButton:hover {{ background: {p.hover}; }}
QPushButton:pressed {{ background: {p.line}; }}
QPushButton:disabled {{ color: {p.text_faint}; border-color: {p.line}; }}
QPushButton[variant="primary"] {{
    background: {p.accent}; border-color: {p.accent}; color: {p.accent_text}; font-weight: 600;
}}
QPushButton[variant="primary"]:hover {{ background: {p.accent_hover}; border-color: {p.accent_hover}; }}
QPushButton[variant="ghost"] {{ background: transparent; border-color: transparent; color: {p.text_muted}; }}
QPushButton[variant="ghost"]:hover {{ color: {p.text}; background: {p.hover}; }}

QToolButton {{
    background: transparent; border: 1px solid transparent; border-radius: {r}px; padding: 4px;
}}
QToolButton:hover {{ background: {p.hover}; }}
QToolButton:pressed {{ background: {p.line}; }}
QToolButton:checked {{ background: {p.raised}; border-color: {p.line_strong}; }}
QToolButton[variant="segment"] {{
    border: 1px solid {p.line_strong}; border-radius: 0; padding: 4px 10px; color: {p.text_muted};
}}
QToolButton[variant="segment"][position="first"] {{ border-top-left-radius: {r}px; border-bottom-left-radius: {r}px; }}
QToolButton[variant="segment"][position="last"] {{ border-top-right-radius: {r}px; border-bottom-right-radius: {r}px; }}
QToolButton[variant="segment"]:checked {{ background: {p.hover}; color: {p.text}; border-color: {p.line_strong}; }}

QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
    background: {p.background};
    border: 1px solid {p.line};
    border-radius: {r}px;
    padding: 3px 6px;
    selection-background-color: {p.accent};
    selection-color: {p.accent_text};
}}
QDoubleSpinBox, QSpinBox {{ font-family: "{TYPE.mono}"; }}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{ border-color: {p.accent}; }}
QLineEdit:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{ color: {p.text_faint}; }}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button, QSpinBox::down-button {{ width: 0; border: none; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox::down-arrow {{ image: none; width: 0; }}
QComboBox QAbstractItemView {{
    background: {p.raised}; border: 1px solid {p.line_strong};
    selection-background-color: {p.hover}; selection-color: {p.text}; padding: 2px;
}}

QTabBar {{ qproperty-drawBase: 0; }}
QTabBar::tab {{
    background: transparent; color: {p.text_muted};
    padding: 7px 14px; border: none; border-bottom: 2px solid transparent;
}}
QTabBar::tab:hover {{ color: {p.text}; }}
QTabBar::tab:selected {{ color: {p.text}; border-bottom-color: {p.accent}; }}
QTabWidget::pane {{ border: none; }}

QMenuBar {{ background: {p.surface}; border-bottom: 1px solid {p.line}; padding: 2px 4px; }}
QMenuBar::item {{ padding: 4px 10px; border-radius: {r}px; }}
QMenuBar::item:selected {{ background: {p.hover}; }}
QMenu {{ background: {p.raised}; border: 1px solid {p.line_strong}; padding: 4px; }}
QMenu::item {{ padding: 5px 24px 5px 10px; border-radius: {r}px; }}
QMenu::item:selected {{ background: {p.hover}; }}
QMenu::item:disabled {{ color: {p.text_faint}; }}
QMenu::separator {{ height: 1px; background: {p.line}; margin: 4px 6px; }}
QMenu::indicator {{ width: 12px; height: 12px; margin-left: 4px; }}

QListView, QTreeView, QTableView {{
    background: {p.surface}; border: none; alternate-background-color: {p.background};
    selection-background-color: {p.hover}; selection-color: {p.text};
}}
QListView::item, QTreeView::item {{ padding: 4px 6px; border-left: 2px solid transparent; }}
QListView::item:selected, QTreeView::item:selected {{ background: {p.hover}; border-left-color: {p.accent}; }}
QHeaderView::section {{
    background: {p.surface}; color: {p.text_muted}; border: none; border-bottom: 1px solid {p.line};
    padding: 4px 6px; font-size: {TYPE.size_small}px;
}}

QScrollBar:vertical {{ background: transparent; width: 9px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 9px; margin: 0; }}
QScrollBar::handle {{ background: {p.line_strong}; border-radius: 4px; min-height: 24px; min-width: 24px; }}
QScrollBar::handle:hover {{ background: {p.text_faint}; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{
    background: none; border: none; width: 0; height: 0;
}}

QSplitter::handle {{ background: {p.line}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}

QSlider::groove:horizontal {{ height: 2px; background: {p.line_strong}; }}
QSlider::sub-page:horizontal {{ background: {p.accent}; }}
QSlider::handle:horizontal {{
    width: 10px; height: 10px; margin: -4px 0; border-radius: 5px; background: {p.text};
}}

QToolTip {{
    background: {p.raised}; color: {p.text}; border: 1px solid {p.line_strong}; padding: 4px 6px;
}}
QProgressBar {{ background: {p.background}; border: 1px solid {p.line}; border-radius: {r}px; height: 6px; text-align: center; }}
QProgressBar::chunk {{ background: {p.accent}; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px; border: 1px solid {p.line_strong}; border-radius: 2px; background: {p.background};
}}
QCheckBox::indicator:checked {{ background: {p.accent}; border-color: {p.accent}; }}
"""
