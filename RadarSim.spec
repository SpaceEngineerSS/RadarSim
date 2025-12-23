# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Collect data files for critical dependencies
datas = []

# Collect sklearn data files if available
try:
    datas += collect_data_files('sklearn')
except Exception:
    pass

# Manual data collection for project assets
project_assets = ['scenarios', 'resources', 'models', 'data']
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
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'pyqtgraph',
        'numpy',
        'scipy',
        'pandas',
        'yaml',
        'matplotlib',
        'numba',
        'h5py',
        'psutil',
        'tqdm',
        'joblib',
        'sklearn',
        'sklearn.utils._cython_blas',
        'sklearn.utils._typedefs',
        'sklearn.neighbors.typedefs',
        'sklearn.neighbors.quad_tree',
        'sklearn.tree._utils',
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
