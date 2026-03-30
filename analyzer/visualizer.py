#!/usr/bin/env python3
"""Standalone local web server for the Project Kestrel visualizer (supersedes backend/editor_bridge.py).

Features:
 - Serves the existing visualizer.html (and any static assets in the folder).
 - Exposes legacy HTTP API endpoints for compatibility while the desktop
    pywebview bridge remains the primary integration path.
 - Intended to be frozen into a single executable with PyInstaller.

Usage (development):
    python visualizer/visualizer.py --port 8765 --root C:/Photos/Trip

After starting it will open the desktop UI (pywebview) at http://127.0.0.1:<port>/ .

Build single-file EXE (example):
    pyinstaller --onefile --name kestrel_viz visualizer/visualizer.py

Optionally set env vars (same as editor_bridge):
        KESTREL_ALLOWED_ROOT=C:/Photos/Trip  (restrict paths)
    KESTREL_BRIDGE_TOKEN=secret              (require auth header)
    KESTREL_ALLOWED_EXTENSIONS=.cr3,.jpg,... (override allowed list)
    KESTREL_ALLOW_ANY_EXTENSION=1            (disable extension filtering)
    KESTREL_ENABLE_LEGACY_HTTP_API=1         (re-enable legacy local HTTP control routes)
    KESTREL_ENABLE_LEGACY_OPEN_ENDPOINT=1    (re-enable legacy HTTP /open endpoint)

"""

from __future__ import annotations

WEBVIEW_IMPORT_SUCCESS = False
try:
    import webview  # type: ignore
    WEBVIEW_IMPORT_SUCCESS = True
except Exception:
    pass

import argparse
import json
import os
import sys

import secrets
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import threading
from urllib.parse import urlparse
from datetime import datetime
from typing import Optional, Set, TextIO

# --- Extracted modules ---
from settings_utils import load_persisted_settings, save_persisted_settings, log, _normalize
from editor_launch import launch
from queue_manager import _queue_manager
from api_bridge import Api

# Telemetry — failsafe import (never blocks startup)
try:
    import kestrel_telemetry as _telemetry
except ImportError:
    try:
        from analyzer import kestrel_telemetry as _telemetry
    except ImportError:
        _telemetry = None  # type: ignore[assignment]

HOST = '127.0.0.1'

# Phase 0 policy lock: desktop pywebview mode is required and API routes use
# a unified token+origin security policy while the local HTTP bridge exists.
SECURITY_POLICY_VERSION = '2026-03-30'
BROWSER_ONLY_MODE_SUPPORTED = False
API_AUTH_POLICY = 'desktop-api-preferred;legacy-http-api-disabled-by-default'
LEGACY_HTTP_API_ENABLED = os.environ.get('KESTREL_ENABLE_LEGACY_HTTP_API') == '1'
LEGACY_OPEN_ENDPOINT_ENABLED = os.environ.get('KESTREL_ENABLE_LEGACY_OPEN_ENDPOINT') == '1'
LEGAL_SELF_HEAL_MIGRATION_KEY = 'legal_upgrade_self_heal_2026_03'

# --- Security / behavior configuration (env override matches editor_bridge) ---
ALLOWED_ROOT = os.environ.get('KESTREL_ALLOWED_ROOT')
if ALLOWED_ROOT:
    ALLOWED_ROOT = os.path.abspath(os.path.expanduser(ALLOWED_ROOT))

AUTH_TOKEN = os.environ.get('KESTREL_BRIDGE_TOKEN')
if not AUTH_TOKEN:
    # Generate an ephemeral token per run for legacy HTTP API compatibility.
    AUTH_TOKEN = secrets.token_urlsafe(32)
MAX_REQUEST_BYTES = int(os.environ.get('KESTREL_MAX_REQUEST_BYTES', '4096'))
ALLOWED_EDITORS: Set[str] = {
    'system', 'darktable', 'lightroom', 'photoshop', 'capture_one',
    'affinity', 'gimp', 'rawtherapee', 'luminar', 'dxo', 'on1',
    'acdsee', 'paintshop', 'faststone', 'xnview', 'irfanview', 'custom',
}
_default_exts = ['.cr3', '.cr2', '.nef', '.arw', '.dng', '.raf', '.orf', '.rw2', '.sr2', '.jpg', '.jpeg', '.png', '.tif', '.tiff']
ALLOWED_EXTENSIONS: Set[str] = set(os.environ.get('KESTREL_ALLOWED_EXTENSIONS', ','.join(_default_exts)).lower().split(','))
ALLOW_ANY_EXTENSION = os.environ.get('KESTREL_ALLOW_ANY_EXTENSION') == '1'

_RUNTIME_LOG_HANDLE: Optional[TextIO] = None


class _TeeStream:
    """Mirror writes to the original stream and a runtime log file."""

    def __init__(self, original_stream, log_handle: TextIO):
        self._original_stream = original_stream
        self._log_handle = log_handle
        self.encoding = getattr(original_stream, 'encoding', 'utf-8')
        self.errors = getattr(original_stream, 'errors', 'replace')

    def write(self, data):
        text = data if isinstance(data, str) else str(data)
        try:
            self._original_stream.write(text)
        except Exception:
            pass
        try:
            self._log_handle.write(text)
        except Exception:
            pass
        return len(text)

    def flush(self):
        try:
            self._original_stream.flush()
        except Exception:
            pass
        try:
            self._log_handle.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return self._original_stream.isatty()
        except Exception:
            return False

    def fileno(self):
        return self._original_stream.fileno()

    @property
    def buffer(self):
        return getattr(self._original_stream, 'buffer', None)

    def __getattr__(self, name):
        return getattr(self._original_stream, name)


def _enable_runtime_log_capture() -> str:
    """Capture process stdout/stderr to a persistent runtime log file."""
    global _RUNTIME_LOG_HANDLE
    try:
        try:
            from kestrel_analyzer.logging_utils import resolve_log_dir
        except ImportError:
            from analyzer.kestrel_analyzer.logging_utils import resolve_log_dir

        base_log_dir = resolve_log_dir(None)
    except Exception:
        base_log_dir = os.path.join(os.path.expanduser('~'), '.kestrel')

    try:
        runtime_dir = os.path.join(base_log_dir, 'logs')
        os.makedirs(runtime_dir, exist_ok=True)
        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        runtime_log_path = os.path.join(runtime_dir, f'kestrel_runtime_{ts}.log')

        _RUNTIME_LOG_HANDLE = open(runtime_log_path, 'a', encoding='utf-8', buffering=1)
        sys.stdout = _TeeStream(sys.stdout, _RUNTIME_LOG_HANDLE)
        sys.stderr = _TeeStream(sys.stderr, _RUNTIME_LOG_HANDLE)
        return runtime_log_path
    except Exception:
        return ''


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + 'Z'


def _mark_session_start() -> None:
    """Mark this app session as active and detect unclean prior shutdown."""
    try:
        settings = load_persisted_settings()
        prev_started = str(settings.get('app_session_started_utc', '') or '').strip()
        prev_clean = bool(settings.get('app_session_closed_cleanly', True))
        if prev_started and not prev_clean:
            settings['last_unclean_shutdown_utc'] = prev_started
        settings['app_session_started_utc'] = _utc_now_iso()
        settings['app_session_closed_cleanly'] = False
        settings['app_session_pid'] = int(os.getpid())
        save_persisted_settings(settings)
    except Exception:
        pass


def _mark_session_clean_exit() -> None:
    """Mark this app session as closed cleanly."""
    try:
        settings = load_persisted_settings()
        settings['app_session_closed_cleanly'] = True
        settings['last_session_closed_utc'] = _utc_now_iso()
        save_persisted_settings(settings)
    except Exception:
        pass


def _apply_legal_upgrade_self_heal(settings: dict, prev_version: str, current_version: str) -> bool:
    """One-time migration for legacy installs that lost legal consent markers.

    Returns True when the migration marker is updated in the settings payload.
    """
    if not isinstance(settings, dict):
        return False
    prev = str(prev_version or '').strip()
    curr = str(current_version or '').strip()
    if not prev or not curr or prev == curr:
        return False
    if settings.get(LEGAL_SELF_HEAL_MIGRATION_KEY, False):
        return False

    legal_agreed = str(settings.get('legal_agreed_version', '') or '').strip()
    if not legal_agreed:
        settings['legal_agreed_version'] = prev or curr
        if 'installed_telemetry_sent' not in settings:
            settings['installed_telemetry_sent'] = True
        log('[legal] Applied one-time upgrade self-heal for missing consent markers:', prev, '->', curr)

    settings[LEGAL_SELF_HEAL_MIGRATION_KEY] = True
    return True


def build_original_path(root: str, rel: str) -> str:
    if ALLOWED_ROOT:
        root = ALLOWED_ROOT
    else:
        root = _normalize(root) if root else ''
    rel = _normalize(rel) if rel else ''
    if not rel or os.path.isabs(rel):
        return ''
    base = os.path.join(root, rel) if root else rel
    return os.path.abspath(base)


def _is_within_root(path: str) -> bool:
    if not path:
        return False
    if not ALLOWED_ROOT:
        return True
    try:
        common = os.path.commonpath([os.path.realpath(path), os.path.realpath(ALLOWED_ROOT)])
        return common == os.path.realpath(ALLOWED_ROOT)
    except Exception:
        return False


def _extension_allowed(path: str) -> bool:
    if ALLOW_ANY_EXTENSION:
        return True
    _, ext = os.path.splitext(path)
    return ext.lower() in ALLOWED_EXTENSIONS


class Handler(SimpleHTTPRequestHandler):
    # Serve from directory of this script (project root) by default.
    def translate_path(self, path: str) -> str:  # type: ignore[override]
        """Resolve file paths robustly across dev, frozen, and installed builds.
        
        Checks multiple locations:
        1. Normal CWD-relative translation (for dev mode)
        2. analyzer/ subfolder (for files like culling.html)
        3. _internal/analyzer/ (for PyInstaller frozen install in Program Files)
        """
        # Try the normal translation first
        resolved = super().translate_path(path)
        if os.path.exists(resolved):
            return resolved
        
        # If not found and path doesn't already contain /analyzer, try analyzer/ prefix
        if not path.startswith('/analyzer'):
            alt = super().translate_path('/analyzer' + path)
            if os.path.exists(alt):
                return alt
        
        # For frozen builds, also check _internal subdirectories
        if getattr(sys, 'frozen', False):
            # Try <exe_dir>/_internal/analyzer/<file>
            try:
                exe_dir = os.path.dirname(sys.executable)
                internal_dir = os.path.join(exe_dir, '_internal')
                alt_path = path.lstrip('/')
                alt = os.path.join(internal_dir, alt_path)
                if os.path.exists(alt):
                    return alt
                # If path already has /analyzer, also check _internal/analyzer/<file>
                if path.startswith('/analyzer'):
                    alt_path = path[1:]  # Strip leading /
                    alt = os.path.join(internal_dir, alt_path)
                    if os.path.exists(alt):
                        return alt
            except Exception:
                pass
            
            # Try _MEIPASS (PyInstaller temp extraction)
            meipass = getattr(sys, '_MEIPASS', None)
            if meipass:
                alt_path = path.lstrip('/')
                alt = os.path.join(meipass, alt_path)
                if os.path.exists(alt):
                    return alt
        
        # Return the original resolution (will 404 if file doesn't exist)
        return resolved

    def end_headers(self):  # Inject basic headers (no wildcard CORS; same-origin only)
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def do_GET(self):  # type: ignore[override]
        # Legacy bridge config endpoint kept for compatibility (no token export).
        if self.path == '/bridge_config.js':
            body = (
                f"// bridge_config.js is deprecated in desktop-only mode\n"
                f"window.__BRIDGE_ORIGIN=window.location.origin;\n"
            ).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/javascript')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == '/settings':
            if not LEGACY_HTTP_API_ENABLED:
                self._json(410, {'ok': False, 'error': 'Legacy HTTP API disabled; use pywebview API.'})
                return
            if not self._check_auth():
                return
            self._json(200, {'ok': True, 'settings': load_persisted_settings()})
            return
        if self.path == '/queue/status':
            if not LEGACY_HTTP_API_ENABLED:
                self._json(410, {'ok': False, 'error': 'Legacy HTTP API disabled; use pywebview API.'})
                return
            if not self._check_auth():
                return
            self._json(200, _queue_manager.get_status())
            return
        if self.path == '/recovery/status':
            if not LEGACY_HTTP_API_ENABLED:
                self._json(410, {'ok': False, 'error': 'Legacy HTTP API disabled; use pywebview API.'})
                return
            if not self._check_auth():
                return
            self.handle_recovery_status()
            return
        if self.path in ('/', '/index.html'):
            # Prefer analyzer/visualizer.html when present (merged layout).
            # Check multiple locations across dev, frozen, and installed builds.
            def _find_visualizer():
                # List of relative paths to try (from various base dirs)
                candidates = [
                    'analyzer/visualizer.html',
                    'visualizer.html',
                ]
                
                # Check from CWD
                for rel in candidates:
                    full = os.path.join(os.getcwd(), rel)
                    if os.path.exists(full):
                        return '/' + rel
                
                # Check from exe dir (frozen/installed)
                try:
                    exe_dir = os.path.dirname(sys.executable)
                    internal_dir = os.path.join(exe_dir, '_internal')
                    for rel in candidates:
                        full = os.path.join(internal_dir, rel)
                        if os.path.exists(full):
                            return '/' + rel
                except Exception:
                    pass
                
                # Check PyInstaller _MEIPASS
                meipass = getattr(sys, '_MEIPASS', None)
                if meipass:
                    for rel in candidates:
                        full = os.path.join(meipass, rel)
                        if os.path.exists(full):
                            return '/' + rel
                
                # Default fallback
                return '/analyzer/visualizer.html'
            
            self.path = _find_visualizer()
        return super().do_GET()

    def do_OPTIONS(self):  # Minimal preflight (only allow same-origin JS; token still required)
        if not (LEGACY_HTTP_API_ENABLED or LEGACY_OPEN_ENDPOINT_ENABLED):
            self.send_response(410)
            self.end_headers()
            return
        origin = self.headers.get('Origin')
        if origin and origin != f'http://{HOST}:{self.server.server_port}':  # type: ignore[attr-defined]
            self.send_response(403); self.end_headers(); return
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', origin or f'http://{HOST}:{self.server.server_port}')  # type: ignore[attr-defined]
        self.send_header('Access-Control-Allow-Headers', 'Content-Type,X-Bridge-Token')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.end_headers()

    def do_POST(self):  # type: ignore[override]
        parsed = urlparse(self.path)
        if parsed.path == '/open':
            if LEGACY_OPEN_ENDPOINT_ENABLED or LEGACY_HTTP_API_ENABLED:
                self.handle_open()
            else:
                self._json(410, {'ok': False, 'error': 'HTTP /open is disabled; use pywebview open_in_editor API.'})
        elif parsed.path == '/settings':
            if not LEGACY_HTTP_API_ENABLED:
                self._json(410, {'ok': False, 'error': 'Legacy HTTP API disabled; use pywebview API.'})
                return
            self.handle_settings()
        elif parsed.path == '/feedback':
            if not LEGACY_HTTP_API_ENABLED:
                self._json(410, {'ok': False, 'error': 'Legacy HTTP API disabled; use pywebview API.'})
                return
            self.handle_feedback()
        elif parsed.path == '/shutdown':
            if not LEGACY_HTTP_API_ENABLED:
                self._json(410, {'ok': False, 'error': 'Legacy HTTP API disabled; use pywebview API.'})
                return
            self.handle_shutdown()
        elif parsed.path == '/queue/start':
            if not LEGACY_HTTP_API_ENABLED:
                self._json(410, {'ok': False, 'error': 'Legacy HTTP API disabled; use pywebview API.'})
                return
            self.handle_queue_start()
        elif parsed.path in ('/queue/pause', '/queue/resume', '/queue/cancel', '/queue/clear'):
            if not LEGACY_HTTP_API_ENABLED:
                self._json(410, {'ok': False, 'error': 'Legacy HTTP API disabled; use pywebview API.'})
                return
            self.handle_queue_control(parsed.path)
        elif parsed.path == '/recovery/restore':
            if not LEGACY_HTTP_API_ENABLED:
                self._json(410, {'ok': False, 'error': 'Legacy HTTP API disabled; use pywebview API.'})
                return
            self.handle_recovery_restore()
        elif parsed.path == '/recovery/clear':
            if not LEGACY_HTTP_API_ENABLED:
                self._json(410, {'ok': False, 'error': 'Legacy HTTP API disabled; use pywebview API.'})
                return
            self.handle_recovery_clear()
        elif parsed.path == '/recovery/report':
            if not LEGACY_HTTP_API_ENABLED:
                self._json(410, {'ok': False, 'error': 'Legacy HTTP API disabled; use pywebview API.'})
                return
            self.handle_recovery_report()
        else:
            self.send_response(404); self.end_headers(); self.wfile.write(b'{}')

    def _read_json(self):
        length = int(self.headers.get('Content-Length', 0))
        if length > MAX_REQUEST_BYTES:
            raise ValueError('Request too large')
        raw = self.rfile.read(length) if length else b''
        if not raw:
            return {}
        return json.loads(raw.decode('utf-8'))

    def _json(self, status: int, obj):
        body = json.dumps(obj).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        # Only echo same-origin to mitigate unsolicited cross-origin token use
        self.send_header('Access-Control-Allow-Origin', f'http://{HOST}:{self.server.server_port}')  # type: ignore[attr-defined]
        self.end_headers()
        self.wfile.write(body)

    def handle_open(self):
        if not self._check_auth():
            return
        try:
            payload = self._read_json()
        except Exception as e:
            self._json(400, {'ok': False, 'error': str(e)}); return
        log('payload', payload)
        root = payload.get('root')
        rel = payload.get('relative')
        editor = (payload.get('editor') or 'system')
        if isinstance(editor, str):
            editor = editor.strip().lower()
        else:
            editor = 'system'
        if editor not in ALLOWED_EDITORS:
            editor = 'system'
        target = build_original_path(root, rel)
        log('open', editor, root, rel, '->', target)
        if not target:
            self._json(400, {'ok': False, 'error': 'Invalid path'}); return
        if not _is_within_root(target):
            self._json(403, {'ok': False, 'error': 'Path escapes allowed root'}); return
        if not os.path.exists(target):
            self._json(404, {
                'ok': False,
                'error': 'File not found',
                'target': target,
                'hint': 'Ensure Settings -> Local Root points to the folder containing your RAW files.'
            }); return
        if not _extension_allowed(target):
            self._json(415, {'ok': False, 'error': 'Extension not allowed', 'target': target, 'allowed': sorted(ALLOWED_EXTENSIONS)}); return
        try:
            launch(target, editor)
            self._json(200, {'ok': True, 'path': target})
        except Exception as e:
            self._json(500, {'ok': False, 'error': str(e)})

    def handle_shutdown(self):
        if not self._check_auth():
            return
        log('Received shutdown request from client; scheduling server shutdown.')
        # Respond first, then shutdown asynchronously so reply is delivered
        self._json(200, {'ok': True, 'message': 'Shutting down'})
        def _shutdown():
            try:
                # slight delay to let response flush
                import time; time.sleep(0.25)
                self.server.shutdown()
            except Exception as e:  # noqa: BLE001
                log('Error during shutdown:', e)
        threading.Thread(target=_shutdown, daemon=True).start()

    def handle_settings(self):
        if not self._check_auth():
            return
        try:
            payload = self._read_json()
            settings = payload.get('settings') if isinstance(payload, dict) else None
            if not isinstance(settings, dict):
                raise ValueError('Invalid settings payload')
            save_persisted_settings(settings)
            self._json(200, {'ok': True})
        except Exception as e:
            self._json(400, {'ok': False, 'error': str(e)})

    def _check_origin(self) -> bool:
        origin = self.headers.get('Origin')
        expected_origin = f'http://{HOST}:{self.server.server_port}'  # type: ignore[attr-defined]
        # Best-effort origin check: browsers send Origin on CORS/XHR/fetch.
        # We allow missing Origin for compatibility with some local clients.
        if origin and origin != expected_origin:
            self._json(403, {'ok': False, 'error': 'Origin mismatch'})
            return False
        return True

    def _check_auth(self) -> bool:
        """Return True if token and origin checks pass for API routes."""
        if AUTH_TOKEN:
            token = self.headers.get('X-Bridge-Token') or ''
            if token != AUTH_TOKEN:
                self._json(401, {'ok': False, 'error': 'Unauthorized'})
                return False
        return self._check_origin()

    def handle_feedback(self):
        """Accept feedback/bug report submissions (browser-mode fallback)."""
        if not self._check_auth():
            return
        try:
            payload = self._read_json()
            if not isinstance(payload, dict):
                self._json(400, {'ok': False, 'error': 'Invalid payload'}); return
            if _telemetry is None:
                self._json(200, {'ok': True, 'note': 'Telemetry unavailable'}); return
            settings = load_persisted_settings()
            machine_id = _telemetry.get_machine_id(settings)
            log_tail = ''
            if payload.get('include_logs', False):
                active_folder = str(settings.get('active_analysis_path', '') or '').strip()
                log_tail = _telemetry.get_recent_log_tail(folder=active_folder or None, runtime_log_files=3)
            _telemetry.send_feedback(
                report_type=payload.get('type', 'general'),
                description=payload.get('description', ''),
                contact=payload.get('contact', ''),
                screenshot_b64=payload.get('screenshot_b64', ''),
                log_tail=log_tail,
                machine_id=machine_id,
                version=_telemetry._read_version(),
            )
            self._json(200, {'ok': True})
        except Exception as e:
            self._json(400, {'ok': False, 'error': str(e)})

    def handle_queue_start(self):
        if not self._check_auth():
            return
        try:
            payload = self._read_json()
            paths = payload.get('paths', []) if isinstance(payload, dict) else []
            use_gpu = bool(payload.get('use_gpu', True)) if isinstance(payload, dict) else True
            if not isinstance(paths, list):
                self._json(400, {'ok': False, 'error': '"paths" must be a list'}); return
            result = _queue_manager.enqueue(paths, use_gpu=use_gpu)
            self._json(200, {'ok': result['success'], **result})
        except Exception as e:
            self._json(400, {'ok': False, 'error': str(e)})

    def handle_queue_control(self, path: str):
        if not self._check_auth():
            return
        if path == '/queue/pause':
            self._json(200, {'ok': True, **_queue_manager.pause()})
        elif path == '/queue/resume':
            self._json(200, {'ok': True, **_queue_manager.resume()})
        elif path == '/queue/cancel':
            self._json(200, {'ok': True, **_queue_manager.cancel()})
        elif path == '/queue/clear':
            self._json(200, {'ok': True, **_queue_manager.clear_done()})
        else:
            self._json(404, {'ok': False, 'error': 'Not found'})

    def handle_recovery_status(self):
        try:
            settings = load_persisted_settings()
            queue_state = _queue_manager.get_persisted_recovery_state()
            unclean_utc = str(settings.get('last_unclean_shutdown_utc', '') or '').strip()
            self._json(
                200,
                {
                    'ok': True,
                    'success': True,
                    'unclean_shutdown': bool(unclean_utc),
                    'unclean_shutdown_utc': unclean_utc,
                    'queue_recovery': queue_state,
                },
            )
        except Exception as e:
            self._json(400, {'ok': False, 'success': False, 'error': str(e)})

    def handle_recovery_restore(self):
        if not self._check_auth():
            return
        try:
            result = _queue_manager.restore_from_persisted_state()
            self._json(200, {'ok': bool(result.get('success')), **result})
        except Exception as e:
            self._json(400, {'ok': False, 'success': False, 'error': str(e)})

    def handle_recovery_clear(self):
        if not self._check_auth():
            return
        try:
            payload = self._read_json()
            clear_queue_state = True
            if isinstance(payload, dict):
                clear_queue_state = bool(payload.get('clear_queue_state', True))
            settings = load_persisted_settings()
            settings.pop('last_unclean_shutdown_utc', None)
            if clear_queue_state:
                settings.pop('queue_recovery_state', None)
            save_persisted_settings(settings)
            self._json(200, {'ok': True, 'success': True})
        except Exception as e:
            self._json(400, {'ok': False, 'success': False, 'error': str(e)})

    def handle_recovery_report(self):
        if not self._check_auth():
            return
        try:
            if _telemetry is None:
                self._json(200, {'ok': False, 'success': False, 'error': 'Telemetry unavailable'})
                return
            settings = load_persisted_settings()
            machine_id = _telemetry.get_machine_id(settings)
            active_folder = str(settings.get('active_analysis_path', '') or '').strip()
            log_tail = _telemetry.get_recent_log_tail(folder=active_folder or None, runtime_log_files=3)
            _telemetry.send_crash_report(
                exc=None,
                tb_str='Recovered unclean shutdown report requested by user.',
                log_tail=log_tail,
                session_analytics={
                    'recovery_report': True,
                    'active_analysis_path': active_folder,
                },
                machine_id=machine_id,
                version=_telemetry._read_version(),
            )
            self._json(200, {'ok': True, 'success': True})
        except Exception as e:
            self._json(400, {'ok': False, 'success': False, 'error': str(e)})


def parse_args():
    ap = argparse.ArgumentParser(description='Serve Project Kestrel visualizer with local desktop bridge.')
    ap.add_argument('--port', type=int, default=8765, help='Port to listen on (default 8765)')
    ap.add_argument('--root', default='', help='Default root folder for RAW originals (client can override unless KESTREL_ALLOWED_ROOT set)')
    return ap.parse_args()


def main():
    args = parse_args()
    runtime_log_path = _enable_runtime_log_capture()
    if runtime_log_path:
        log('Runtime log capture enabled:', runtime_log_path)
    log('Security policy:', SECURITY_POLICY_VERSION, API_AUTH_POLICY, 'browser_mode_supported=', BROWSER_ONLY_MODE_SUPPORTED)
    log('Legacy HTTP control API enabled =', LEGACY_HTTP_API_ENABLED, '| legacy /open enabled =', LEGACY_OPEN_ENDPOINT_ENABLED)
    _mark_session_start()

    # When visualizer.py is run from inside analyzer/ (merged layout) set
    # the working directory to the repository root so assets and shared
    # files (assets/, visualizer files) are served correctly.
    # If frozen by PyInstaller (onedir), prefer the bundled _internal folder
    # inside the distribution so static assets (visualizer.html, logos) are
    # served from the on-disk bundle.
    if getattr(sys, 'frozen', False):
        meipass = getattr(sys, '_MEIPASS', None) or os.path.dirname(sys.executable)
        candidate = os.path.join(meipass, '_internal')
        if os.path.isdir(candidate):
            os.chdir(candidate)
        elif meipass and os.path.isdir(meipass):
            os.chdir(meipass)
        else:
            # Fallback to repo-root relative when running unpacked
            os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..') or '.')
    else:
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..') or '.')
    server = ThreadingHTTPServer((HOST, args.port), Handler)
    log(f'Serving visualizer at http://{HOST}:{args.port}/  (Press Ctrl+C to stop)')
    if LEGACY_HTTP_API_ENABLED or LEGACY_OPEN_ENDPOINT_ENABLED:
        log('HTTP bridge token initialized for legacy endpoint compatibility.')
    else:
        log('Legacy HTTP control endpoints are disabled by default.')

    # ── Settings init: ensure machine_id and version are persisted ──
    try:
        if _telemetry is not None:
            _init_settings = load_persisted_settings()
            _prev_version = str(_init_settings.get('version', '') or '').strip()
            _current_version = _telemetry._read_version()
            _telemetry.get_machine_id(_init_settings)
            _init_settings['version'] = _current_version
            _init_settings.setdefault('raw_preview_cache_enabled', True)
            _init_settings.setdefault('exposure_compensation_profile', 'normal')
            _init_settings.setdefault('auto_save_enabled', True)

            _apply_legal_upgrade_self_heal(_init_settings, _prev_version, _current_version)

            save_persisted_settings(_init_settings)
    except Exception:
        pass  # failsafe

    if args.root:
        log('Default root (client-supplied):', args.root)
    url = f'http://{HOST}:{args.port}/'
    if not WEBVIEW_IMPORT_SUCCESS:
        raise RuntimeError('pywebview is required. Browser-only mode is no longer supported.')
    log('Windowed mode enabled; using pywebview')

    def _serve():
        try:
            server.serve_forever()
        except Exception:
            pass

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    api = None
    try:
        log('Starting windowed UI via pywebview...')
        api = Api() # start maximized
        api._server_port = args.port
        win = webview.create_window('Project Kestrel', url, js_api=api, maximized=True)
        api._main_window = win

        # When the analysis queue is running, intercept the close event so the
        # window minimizes to the taskbar instead of killing mid-analysis.
        def _on_closing():
            # Check for unsaved changes via the Python-side flag
            # (avoid evaluate_js here - it deadlocks because closing runs on the GUI thread)
            has_unsaved = getattr(api, '_has_unsaved_changes', False)

            def _cleanup_preview_cache_before_exit():
                try:
                    if hasattr(api, 'cleanup_tracked_culling_caches'):
                        api.cleanup_tracked_culling_caches()
                except Exception as e:
                    log('Cache cleanup on close failed:', e)

            # When an analysis is running or paused, prompt the user with
            # options to Minimize, Exit (cancel) or Cancel the close.
            if _queue_manager.is_running or _queue_manager.is_paused:
                try:
                    # Use native Windows MessageBox if available for a simple
                    # three-button prompt. Fallback to tkinter dialog when not.
                    if sys.platform.startswith('win'):
                        import ctypes
                        MB_YESNOCANCEL = 0x00000003
                        MB_ICONQUESTION = 0x00000020
                        title = 'Analysis in progress'
                        if _queue_manager.is_paused:
                            msg = 'Analysis is paused. Exit Project Kestrel? You can re-open later to resume.'
                        else:
                            msg = 'Analysis is in progress. Cancel analysis and exit?'
                        resp = ctypes.windll.user32.MessageBoxW(0, msg, title, MB_YESNOCANCEL | MB_ICONQUESTION)
                        # IDYES=6 -> Exit (cancel analysis and close)
                        # IDNO=7  -> Minimize instead of closing
                        # IDCANCEL=2 -> Do not close
                        if resp == 6:
                            try:
                                _queue_manager.cancel()
                            except Exception:
                                pass
                            _cleanup_preview_cache_before_exit()
                            return True
                        if resp == 7:
                            try:
                                win.minimize()
                            except Exception:
                                pass
                            return False
                        return False
                    else:
                        # Tkinter fallback
                        import tkinter as _tk
                        from tkinter import messagebox as _mb
                        root = _tk.Tk()
                        root.withdraw()
                        if _queue_manager.is_paused:
                            msg = 'Analysis is paused. Exit Project Kestrel? You can re-open later to resume.'
                        else:
                            msg = 'Analysis is in progress. Cancel analysis and exit?'
                        res = _mb.askyesnocancel('Analysis in progress', msg)
                        root.destroy()
                        # askyesnocancel returns True=Yes, False=No, None=Cancel
                        if res is True:
                            try:
                                _queue_manager.cancel()
                            except Exception:
                                pass
                            _cleanup_preview_cache_before_exit()
                            return True
                        if res is False:
                            try:
                                win.minimize()
                            except Exception:
                                pass
                            return False
                        return False
                except Exception:
                    # If the prompt fails, fall back to minimizing when running
                    try:
                        win.minimize()
                    except Exception:
                        pass
                    return False

            # Prompt for unsaved changes when no analysis is running
            if has_unsaved:
                try:
                    if sys.platform.startswith('win'):
                        import ctypes
                        MB_YESNO = 0x00000004
                        MB_ICONWARNING = 0x00000030
                        msg = 'You have unsaved changes that will be lost. Close anyway?'
                        title = 'Unsaved Changes'
                        resp = ctypes.windll.user32.MessageBoxW(0, msg, title, MB_YESNO | MB_ICONWARNING)
                        if resp == 6:  # Yes - close and discard
                            _cleanup_preview_cache_before_exit()
                            return True
                        return False  # No - don't close
                    else:
                        import tkinter as _tk
                        from tkinter import messagebox as _mb
                        root = _tk.Tk()
                        root.withdraw()
                        res = _mb.askyesno(
                            'Unsaved Changes',
                            'You have unsaved changes that will be lost. Close anyway?'
                        )
                        root.destroy()
                        if res:
                            _cleanup_preview_cache_before_exit()
                            return True
                        return False
                except Exception:
                    _cleanup_preview_cache_before_exit()
                    return True  # on failure, allow close

            _cleanup_preview_cache_before_exit()
            return True  # allow normal close

        try:
            win.events.closing += _on_closing
        except Exception:
            pass  # older pywebview versions may not support this event

        webview.start()
    finally:
        try:
            if api is not None and hasattr(api, 'cleanup_tracked_culling_caches'):
                api.cleanup_tracked_culling_caches()
        except Exception as e:
            log('Cache cleanup during shutdown failed:', e)
        server.shutdown()
        server.server_close()
        log('Server stopped.')

    _mark_session_clean_exit()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        _mark_session_clean_exit()
    except Exception as _main_exc:
        # Top-level crash handler — send crash report before re-raising
        try:
            import traceback as _tb
            if _telemetry is not None:
                _crash_settings = load_persisted_settings()
                _crash_mid = _telemetry.get_machine_id(_crash_settings)
                
                # Fetch recent log tail, passing the active folder's log if available
                _folder_path = _crash_settings.get('active_analysis_path', '')
                if _folder_path:
                    _log_tail = _telemetry.get_recent_log_tail(folder=_folder_path, runtime_log_files=3)
                else:
                    _log_tail = _telemetry.get_recent_log_tail(runtime_log_files=3)
                
                _telemetry.send_crash_report(
                    exc=_main_exc,
                    tb_str=_tb.format_exc(),
                    log_tail=_log_tail,
                    machine_id=_crash_mid,
                    version=_telemetry._read_version(),
                )
                # Give daemon thread a moment to fire off the HTTP request
                import time as _t
                _t.sleep(2)
        except Exception:
            pass  # crash handler itself must never hide the real error
        raise
