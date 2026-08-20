# -*- mode: python ; coding: utf-8 -*-

import os

block_cipher = None

datas = []

# Package the built-in scenarios with the desktop application.
project_assets = ['scenarios']
for asset in project_assets:
    if os.path.exists(asset):
        datas.append((asset, asset))

# Add src files manually to be absolutely sure, although Analysis usually does this
# But here we only add non-python files if they ever exist. 
# Based on find_by_name, there are none currently.

a = Analysis(
    ['run_gui.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'pyqtgraph',
        'pyqtgraph.opengl',
        'OpenGL',
        'numpy',
        'scipy',
        'yaml',
        'matplotlib',
        'numba',
        'h5py',
        'src',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RadarSim',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['resources/icon.ico'] if os.path.exists('resources/icon.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RadarSim',
)
