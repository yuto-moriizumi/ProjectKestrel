"""Persisted settings I/O and general utility functions for the Kestrel visualizer."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

SETTINGS_FILENAME = 'settings.json'
_MAX_PATH_CHARS = 4096
_MAX_TEXT_CHARS = 4096

_ALLOWED_EDITORS = {
    'system', 'darktable', 'lightroom', 'photoshop', 'capture_one',
    'affinity', 'gimp', 'rawtherapee', 'luminar', 'dxo', 'on1',
    'acdsee', 'paintshop', 'faststone', 'xnview', 'irfanview', 'custom',
}
_ALLOWED_RATING_PROFILES = {'very_strict', 'strict', 'balanced', 'lenient', 'very_lenient'}
_ALLOWED_EXPOSURE_PROFILES = {'lenient', 'normal', 'aggressive'}
_ALLOWED_EXPOSURE_SOLVERS = {'legacy_iterative', 'two_pass', 'single_pass', 'predictive_fast', 'adaptive_fast'}
_ALLOWED_QUEUE_ITEM_STATUSES = {'pending', 'running', 'done', 'error', 'cancelled'}

# Telemetry — failsafe import (never blocks startup)
try:
    import kestrel_telemetry as _telemetry
except ImportError:
    try:
        from analyzer import kestrel_telemetry as _telemetry
    except ImportError:
        _telemetry = None  # type: ignore[assignment]


def _get_user_data_dir() -> str:
    # Use a unified application folder name for Project Kestrel
    if sys.platform.startswith('win'):
        base = os.environ.get('LOCALAPPDATA') or os.environ.get('APPDATA') or os.path.expanduser('~')
        return os.path.join(base, 'ProjectKestrel')
    if sys.platform == 'darwin':
        return os.path.join(os.path.expanduser('~'), 'Library', 'Application Support', 'ProjectKestrel')
    base = os.environ.get('XDG_DATA_HOME') or os.path.join(os.path.expanduser('~'), '.local', 'share')
    return os.path.join(base, 'project-kestrel')


def _get_settings_path() -> str:
    return os.path.join(_get_user_data_dir(), SETTINGS_FILENAME)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {'1', 'true', 'yes', 'on'}:
            return True
        if low in {'0', 'false', 'no', 'off'}:
            return False
    return bool(default)


def _coerce_optional_bool(value: Any, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    return _coerce_bool(value, default=False)


def _coerce_int(value: Any, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
    try:
        if isinstance(value, bool):
            n = int(value)
        else:
            n = int(float(value))
    except (TypeError, ValueError):
        n = int(default)
    if min_value is not None and n < min_value:
        n = min_value
    if max_value is not None and n > max_value:
        n = max_value
    return n


def _coerce_float(value: Any, default: float, min_value: float | None = None, max_value: float | None = None) -> float:
    try:
        if isinstance(value, bool):
            n = float(int(value))
        else:
            n = float(value)
    except (TypeError, ValueError):
        n = float(default)
    if min_value is not None and n < min_value:
        n = min_value
    if max_value is not None and n > max_value:
        n = max_value
    return n


def _coerce_string(value: Any, default: str = '', max_len: int = _MAX_TEXT_CHARS) -> str:
    if value is None:
        return default
    s = str(value).strip()
    if not s:
        return default
    if len(s) > max_len:
        s = s[:max_len]
    return s


def _coerce_enum(value: Any, allowed: set[str], default: str) -> str:
    s = _coerce_string(value, default=default, max_len=64).lower()
    if s in allowed:
        return s
    return default


def _coerce_path(value: Any, default: str = '') -> str:
    s = _coerce_string(value, default=default, max_len=_MAX_PATH_CHARS)
    if not s:
        return ''
    return _normalize(s)


def _sanitize_path_list(value: Any, max_items: int = 256) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        p = _coerce_path(item)
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
        if len(out) >= max_items:
            break
    return out


def _sanitize_int_list(value: Any, max_items: int = 128, min_value: int = 0, max_value: int = 1_000_000_000) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    seen: set[int] = set()
    for item in value:
        try:
            n = int(float(item))
        except (TypeError, ValueError):
            continue
        if n < min_value or n > max_value or n in seen:
            continue
        seen.add(n)
        out.append(n)
        if len(out) >= max_items:
            break
    return out


def _sanitize_pending_analytics(value: Any) -> dict | None:
    if not isinstance(value, dict):
        return None

    payload: dict[str, Any] = {
        'folder_path': _coerce_path(value.get('folder_path', '')),
        'files_analyzed': _coerce_int(value.get('files_analyzed', 0), 0, min_value=0, max_value=1_000_000_000),
        'total_files': _coerce_int(value.get('total_files', 0), 0, min_value=0, max_value=1_000_000_000),
        'active_compute_time_s': round(
            _coerce_float(value.get('active_compute_time_s', 0.0), 0.0, min_value=0.0, max_value=31_536_000.0),
            3,
        ),
        'was_cancelled': _coerce_bool(value.get('was_cancelled', False), default=False),
        'machine_id': _coerce_string(value.get('machine_id', ''), max_len=128),
        'version': _coerce_string(value.get('version', ''), max_len=64),
    }

    file_sizes: list[float] = []
    if isinstance(value.get('file_sizes_kb'), list):
        for raw in value.get('file_sizes_kb', []):
            try:
                f = float(raw)
            except (TypeError, ValueError):
                continue
            if f < 0:
                continue
            file_sizes.append(round(min(f, 1_000_000_000.0), 3))
            if len(file_sizes) >= 20000:
                break
    payload['file_sizes_kb'] = file_sizes

    file_formats: dict[str, int] = {}
    if isinstance(value.get('file_formats'), dict):
        for raw_k, raw_v in value.get('file_formats', {}).items():
            k = _coerce_string(raw_k, max_len=32).lower()
            if not k:
                continue
            file_formats[k] = _coerce_int(raw_v, 0, min_value=0, max_value=1_000_000_000)
            if len(file_formats) >= 128:
                break
    payload['file_formats'] = file_formats

    return payload


def _sanitize_queue_recovery_state(value: Any) -> dict | None:
    if not isinstance(value, dict):
        return None

    state: dict[str, Any] = {
        'updated_utc': _coerce_string(value.get('updated_utc', ''), max_len=64),
        'running': _coerce_bool(value.get('running', False), default=False),
        'paused': _coerce_bool(value.get('paused', False), default=False),
        'restore_paths': _sanitize_path_list(value.get('restore_paths'), max_items=512),
    }

    opts = value.get('options') if isinstance(value.get('options'), dict) else {}
    state['options'] = {
        'use_gpu': _coerce_bool(opts.get('use_gpu', True), default=True),
        'wildlife_enabled': _coerce_bool(opts.get('wildlife_enabled', True), default=True),
        'detection_threshold': _coerce_float(opts.get('detection_threshold', 0.75), 0.75, min_value=0.1, max_value=0.99),
        'scene_time_threshold': _coerce_float(opts.get('scene_time_threshold', 1.0), 1.0, min_value=0.0, max_value=60.0),
        'mask_threshold': _coerce_float(opts.get('mask_threshold', 0.5), 0.5, min_value=0.5, max_value=0.95),
        'max_bird_crops': _coerce_int(opts.get('max_bird_crops', 5), 5, min_value=1, max_value=20),
    }

    items_out: list[dict[str, Any]] = []
    items_raw = value.get('items') if isinstance(value.get('items'), list) else []
    for entry in items_raw:
        if not isinstance(entry, dict):
            continue
        path = _coerce_path(entry.get('path', ''))
        if not path:
            continue
        status = _coerce_string(entry.get('status', 'pending'), default='pending', max_len=16).lower()
        if status not in _ALLOWED_QUEUE_ITEM_STATUSES:
            status = 'pending'
        items_out.append(
            {
                'path': path,
                'name': _coerce_string(entry.get('name', ''), max_len=512),
                'status': status,
                'processed': _coerce_int(entry.get('processed', 0), 0, min_value=0, max_value=1_000_000_000),
                'total': _coerce_int(entry.get('total', 0), 0, min_value=0, max_value=1_000_000_000),
            }
        )
        if len(items_out) >= 2000:
            break
    state['items'] = items_out
    return state


def _sanitize_settings_payload(data: dict, emit_log: bool = False) -> dict:
    if not isinstance(data, dict):
        return {}

    out: dict[str, Any] = {}

    def _set_bool(key: str, default: bool) -> None:
        if key in data:
            out[key] = _coerce_bool(data.get(key), default=default)

    def _set_opt_bool(key: str, default: bool | None = None) -> None:
        if key in data:
            out[key] = _coerce_optional_bool(data.get(key), default=default)

    def _set_int(key: str, default: int, min_value: int | None = None, max_value: int | None = None) -> None:
        if key in data:
            out[key] = _coerce_int(data.get(key), default=default, min_value=min_value, max_value=max_value)

    def _set_float(key: str, default: float, min_value: float | None = None, max_value: float | None = None, digits: int | None = None) -> None:
        if key in data:
            val = _coerce_float(data.get(key), default=default, min_value=min_value, max_value=max_value)
            out[key] = round(val, digits) if digits is not None else val

    def _set_str(key: str, default: str = '', max_len: int = _MAX_TEXT_CHARS) -> None:
        if key in data:
            out[key] = _coerce_string(data.get(key), default=default, max_len=max_len)

    def _set_path(key: str, default: str = '') -> None:
        if key in data:
            out[key] = _coerce_path(data.get(key), default=default)

    if 'editor' in data:
        out['editor'] = _coerce_enum(data.get('editor'), _ALLOWED_EDITORS, default='darktable')
    _set_path('customEditorPath')
    _set_int('treeScanDepth', default=3, min_value=1, max_value=6)

    _set_opt_bool('analytics_opted_in', default=None)
    _set_bool('analytics_consent_shown', default=False)

    if 'rating_profile' in data:
        out['rating_profile'] = _coerce_enum(data.get('rating_profile'), _ALLOWED_RATING_PROFILES, default='balanced')
    _set_float('detection_threshold', default=0.75, min_value=0.1, max_value=0.99, digits=4)
    _set_float('scene_time_threshold', default=1.0, min_value=0.0, max_value=60.0, digits=4)
    _set_float('mask_threshold', default=0.5, min_value=0.5, max_value=0.95, digits=4)
    _set_int('max_bird_crops', default=5, min_value=1, max_value=20)

    if 'exposure_compensation_profile' in data:
        out['exposure_compensation_profile'] = _coerce_enum(
            data.get('exposure_compensation_profile'),
            _ALLOWED_EXPOSURE_PROFILES,
            default='aggressive',
        )

    if 'exposure_compensation_solver' in data:
        out['exposure_compensation_solver'] = _coerce_enum(
            data.get('exposure_compensation_solver'),
            _ALLOWED_EXPOSURE_SOLVERS,
            default='adaptive_fast',
        )

    _set_bool('raw_preview_cache_enabled', default=True)
    _set_bool('raw_preview_debug_logging_enabled', default=True)
    _set_bool('auto_save_enabled', default=True)
    _set_bool('raw_exposure_correction_disabled', default=False)

    _set_bool('includeSecondarySpecies', default=False)
    _set_bool('groupByFolder', default=True)
    _set_bool('groupByTime', default=True)
    _set_bool('onlyManualRatedScenes', default=False)
    _set_float('scene_preview_split_ratio', default=0.68, min_value=0.25, max_value=0.85, digits=4)

    if 'sortBy' in data:
        sort_by = _coerce_string(data.get('sortBy'), default='captureTime', max_len=64)
        if sort_by and not sort_by.replace('_', '').isalnum():
            sort_by = 'captureTime'
        out['sortBy'] = sort_by

    _set_path('rootHint')
    if 'lastQueueState' in data:
        out['lastQueueState'] = _sanitize_path_list(data.get('lastQueueState'), max_items=512)

    _set_bool('main_tutorial_seen', default=False)
    if 'kestrel_donate_thresholds_shown' in data:
        out['kestrel_donate_thresholds_shown'] = _sanitize_int_list(
            data.get('kestrel_donate_thresholds_shown'),
            max_items=128,
            min_value=0,
            max_value=1_000_000_000,
        )

    _set_int('kestrel_impact_total_files', default=0, min_value=0, max_value=1_000_000_000_000)
    _set_float('kestrel_impact_total_seconds', default=0.0, min_value=0.0, max_value=31_536_000_000.0, digits=1)

    _set_str('machine_id', max_len=128)
    _set_str('version', max_len=64)
    _set_str('legal_agreed_version', max_len=64)
    _set_bool('installed_telemetry_sent', default=False)
    _set_bool('legal_upgrade_self_heal_2026_03', default=False)

    _set_path('active_analysis_path')
    _set_str('app_session_started_utc', max_len=64)
    _set_bool('app_session_closed_cleanly', default=True)
    _set_int('app_session_pid', default=0, min_value=0, max_value=2_147_483_647)
    _set_str('last_session_closed_utc', max_len=64)
    _set_str('last_unclean_shutdown_utc', max_len=64)

    if 'queue_recovery_state' in data:
        queue_state = _sanitize_queue_recovery_state(data.get('queue_recovery_state'))
        if queue_state is not None:
            out['queue_recovery_state'] = queue_state

    if 'pending_analytics' in data:
        pending = _sanitize_pending_analytics(data.get('pending_analytics'))
        if pending is not None:
            out['pending_analytics'] = pending

    if emit_log:
        unknown = sorted([str(k) for k in data.keys() if k not in out])
        if unknown:
            sample = ', '.join(unknown[:12])
            suffix = ' ...' if len(unknown) > 12 else ''
            log(f'[settings] Dropped unsupported keys ({len(unknown)}): {sample}{suffix}')

    return out


def load_persisted_settings() -> dict:
    path = _get_settings_path()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return _sanitize_settings_payload(data) if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def save_persisted_settings(data: dict) -> None:
    if not isinstance(data, dict):
        raise ValueError('Settings payload must be an object')
    data = _sanitize_settings_payload(data, emit_log=True)
        
    # --- Flush pending analytics on consent ---
    if data.get('analytics_consent_shown', False) and 'pending_analytics' in data:
        pending = data.pop('pending_analytics')
        if data.get('analytics_opted_in', False) and _telemetry is not None:
            try:
                _telemetry.send_folder_analytics(**pending)
                log('[analytics] Flushed pending detailed analytics after opt-in.')
            except Exception as e:
                log(f'[analytics] Failed to flush pending analytics: {e}')
    # ------------------------------------------

    path = _get_settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    try:
        # Remove any stale .tmp from a previous crash (Windows can't replace a locked file)
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        # Fallback: write directly — non-atomic but safe for settings
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, sort_keys=True)
        except OSError as e:
            print(f'[settings] Failed to save settings: {e}', file=sys.stderr)


def log(*args):
    print('[serve]', *args, file=sys.stderr)


def _normalize(p: str) -> str:
    if not p:
        return ''
    p = os.path.expanduser(p)
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1]
    return os.path.normpath(p)
