"""Host-owned, durable bounds on successful Meta generations in one round.

Only Meta traffic uses this local proxy. Every request still passes through the
existing shared monetary gate; the proxy has no real provider credential.
"""
from __future__ import annotations

import contextlib
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import sqlite3
import threading
import time
from urllib.parse import urlsplit
import uuid

from rpent.utils.execution import file_lock, process_identity, owns_process
from meta_harness.pool import replace_record

CAP_CODE = "meta_round_generation_cap"
CLOSING_PROMPT = (
    "The host has reached this round's development generation limit. Continue "
    "the SAME conversation and workspace, but only finalize and submit the best "
    "existing candidate now. Do not start a new hypothesis, live scene, or "
    "evaluation. At most two finalization model generations remain. Pending "
    "evaluations retain their original IDs. Submit remains available."
)


def historical_calls(store, context, stop_after=34):
    """Count distinct positive usage updates, not repeated rate-limit events."""
    key = context['scope_id'].replace('/', '-')
    prefix = f"{key}-development-r{context['round_index']:02d}-d"
    threads = set()
    attempts=store.path('meta/attempts')
    for folder in attempts.iterdir() if attempts.exists() else ():
        if folder.name.startswith(prefix):
            try:
                sid = json.loads((folder/'session.json').read_text()).get('thread_id')
                if sid: threads.add(sid)
            except (OSError, ValueError): pass
    db = store.path(f"workspaces/{context['scope_id']}/runtime/state_5.sqlite")
    if not threads or not db.exists(): return 0
    count = 0
    with sqlite3.connect(f'file:{db}?mode=ro', uri=True, timeout=5) as conn:
        for sid in sorted(threads):
            row = conn.execute('SELECT rollout_path FROM threads WHERE id=?', (sid,)).fetchone()
            if not row or not row[0]: continue
            previous = 0
            try:
                with open(row[0], 'rb') as stream:
                    for line in stream:
                        if b'"token_count"' not in line: continue
                        try:
                            event = json.loads(line); payload = event.get('payload', {})
                            if payload.get('type') != 'token_count': continue
                            info = payload.get('info') or {}
                            total = int((info.get('total_token_usage') or {}).get('total_tokens', 0))
                        except (ValueError, TypeError): continue
                        if total > previous:
                            count += 1; previous = total
                            if count >= stop_after: return count
            except OSError: pass
    return count


class RoundLedger:
    def __init__(self, store, context, limits, historical=0):
        self.store = store
        identity = [str(store.root.resolve()), context['scope_id'], context['round_index']]
        self.key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        self.prefix = f'meta/round-generation-limits/{self.key}'
        self.path = store.path(self.prefix+'.json')
        self.lock = store.path(self.prefix+'.lock')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.limits = (limits['development_requests'], limits['finalization_requests'])
        if not (1 <= self.limits[0] <= 32 and 1 <= self.limits[1] <= 2):
            raise ValueError('Round limits cannot exceed32 development and2 finalization requests')
        with file_lock(self.lock):
            if self.path.exists():
                state = self.read()
                if tuple(state['limits']) != self.limits:
                    raise ValueError('A restored round cannot obtain a different grant')
                # A dead proxy may have delivered a response before persisting its
                # completion. Conservatively consume its outstanding permit.
                for rid, item in list(state['pending'].items()):
                    if not owns_process(item['owner']):
                        state[item['phase']+'_used'] += 1
                        state.setdefault('uncertain_after_worker_exit', []).append(rid)
                        del state['pending'][rid]
                state['development_used']=max(state['development_used'],min(historical,self.limits[0]))
                state['finalization_used']=max(state['finalization_used'],
                    min(max(0,historical-self.limits[0]),self.limits[1]))
            else:
                state = {'identity': identity, 'limits': self.limits,
                    'development_used': min(historical, self.limits[0]),
                    'finalization_used': min(max(0, historical-self.limits[0]), self.limits[1]),
                    'historical_observed_at_least': historical,
                    'pending': {}, 'closing_acknowledged': False, 'created_at': time.time()}
            self.save(state)

    def read(self): return json.loads(self.path.read_text())
    def save(self, state): replace_record(self.store, self.prefix+'.json', state)
    def phase(self):
        state = self.read()
        if state['development_used'] < self.limits[0]: return 'development'
        return 'finalization' if state['finalization_used'] < self.limits[1] else 'exhausted'
    def acknowledge(self):
        with file_lock(self.lock):
            state=self.read();state['closing_acknowledged']=True;self.save(state)
    def reserve(self):
        with file_lock(self.lock):
            state=self.read()
            phase = ('development' if state['development_used'] < self.limits[0]
                     else 'finalization')
            limit = self.limits[0 if phase == 'development' else 1]
            pending = sum(p['phase']==phase for p in state['pending'].values())
            if phase=='finalization' and not state['closing_acknowledged']: return None
            if state[phase+'_used']+pending >= limit: return None
            rid=uuid.uuid4().hex
            state['pending'][rid]={'phase':phase,'owner':process_identity(),'at':time.time()}
            self.save(state);return rid
    def finish(self, rid, generated):
        with file_lock(self.lock):
            state=self.read();item=state['pending'].pop(rid, None)
            if item and generated: state[item['phase']+'_used']+=1
            self.save(state)


class Completion:
    def __init__(self): self.buffer=b'';self.generated=False
    def payload(self, value):
        if not isinstance(value, dict): return
        response=value.get('response',value)
        if value.get('type')=='response.completed' or response.get('status')=='completed':
            self.generated=True
        # Paid partial/failed responses with usable usage consume the same grant.
        usage=response.get('usage') or value.get('usage') or {}
        if usage.get('output_tokens',usage.get('completion_tokens',0)) > 0: self.generated=True
    def feed(self, chunk):
        self.buffer+=chunk
        while b'\n' in self.buffer:
            line,self.buffer=self.buffer.split(b'\n',1)
            if line.startswith(b'data:'):
                with contextlib.suppress(ValueError, UnicodeDecodeError): self.payload(json.loads(line[5:]))


class RoundProxy:
    def __init__(self, attempt, provider, limits):
        self.attempt=attempt; self.provider=provider
        self.ledger=RoundLedger(attempt.store, attempt.context, limits,
            historical_calls(attempt.store,attempt.context))
        self.client_key=secrets.token_urlsafe(32)
        self.upstream_key=os.environ[provider['env_key']]
        target=urlsplit(provider['base_url'])
        if target.scheme!='http' or target.hostname!='127.0.0.1' or target.port!=18445:
            raise ValueError('Round proxy must use the existing shared monetary gate')
        proxy=self
        class Handler(BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'
            def log_message(self,*args): pass
            def reply(self,status,value):
                raw=json.dumps(value).encode()
                self.send_response(status);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(raw)));self.end_headers()
                with contextlib.suppress(OSError): self.wfile.write(raw)
            def do_POST(self):
                if not secrets.compare_digest(self.headers.get('Authorization',''), 'Bearer '+proxy.client_key):
                    return self.reply(401,{'error':{'code':'invalid_round_key'}})
                if self.path!='/v1/responses':return self.reply(404,{'error':{'code':'unsupported_round_route'}})
                try: size=int(self.headers.get('Content-Length','0'))
                except ValueError:return self.reply(400,{'error':{'code':'invalid_request'}})
                if not 0<size<=64*1024*1024:return self.reply(413,{'error':{'code':'invalid_request_size'}})
                body=self.rfile.read(size)
                rid=proxy.ledger.reserve()
                if rid is None:
                    if proxy.ledger.phase()=='development':
                        return self.reply(503,{'error':{'code':'round_request_pending','retryable':True}})
                    return self.reply(409,{'error':{'code':CAP_CODE,'message':CLOSING_PROMPT,
                        'phase':proxy.ledger.phase(),'retryable':False}})
                capture=Completion();connection=None;forwarded=False
                try:
                    connection=http.client.HTTPConnection(target.hostname,target.port,timeout=None)
                    headers={k:v for k,v in self.headers.items() if k.lower() not in
                        {'authorization','host','content-length','connection','transfer-encoding','accept-encoding'}}
                    headers.update({'Authorization':'Bearer '+proxy.upstream_key,'Content-Type':'application/json'})
                    connection.request('POST',target.path.rstrip('/')+'/responses',body,headers)
                    response=connection.getresponse()
                    is_sse='text/event-stream' in response.getheader('Content-Type','')
                    if not is_sse:
                        data=response.read()
                        if response.status<400:
                            with contextlib.suppress(ValueError,UnicodeDecodeError):capture.payload(json.loads(data))
                        proxy.ledger.finish(rid,capture.generated);rid=None
                        self.send_response(response.status)
                        self.send_header('Content-Type',response.getheader('Content-Type','application/json'))
                        self.send_header('Content-Length',str(len(data)));self.end_headers();forwarded=True
                        with contextlib.suppress(OSError):self.wfile.write(data)
                    else:
                        self.send_response(response.status);self.send_header('Content-Type','text/event-stream')
                        self.send_header('Transfer-Encoding','chunked');self.end_headers();forwarded=True
                        disconnected=False
                        while True:
                            chunk=response.read1(65536)
                            if not chunk:break
                            capture.feed(chunk)
                            if not disconnected:
                                try:self.wfile.write(f'{len(chunk):X}\r\n'.encode()+chunk+b'\r\n');self.wfile.flush()
                                except OSError:disconnected=True;self.close_connection=True
                        proxy.ledger.finish(rid,capture.generated);rid=None
                        if not disconnected:
                            with contextlib.suppress(OSError):self.wfile.write(b'0\r\n\r\n');self.wfile.flush()
                except (OSError,http.client.HTTPException):
                    if forwarded:self.close_connection=True
                    else:self.reply(503,{'error':{'code':'round_proxy_transport_wait','retryable':True}})
                finally:
                    if rid is not None:proxy.ledger.finish(rid,capture.generated)
                    if connection:connection.close()
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.server.daemon_threads=True
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()

    def provider_config(self):
        return {**self.provider,'base_url':f'http://127.0.0.1:{self.server.server_port}/v1'}
    def close(self): self.server.shutdown();self.server.server_close()


def configured_limits(attempt):
    from meta_harness.campaign_runtime import control
    return control(attempt.store).get('meta_round_generation_limit')


def phase_for(attempt):
    if not configured_limits(attempt):return 'development'
    identity=[str(attempt.store.root.resolve()),attempt.context['scope_id'],attempt.context['round_index']]
    key=hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    path=attempt.store.path(f'meta/round-generation-limits/{key}.json')
    if not path.exists():return 'development'
    state=json.loads(path.read_text())
    if state['development_used']<state['limits'][0]:return 'development'
    return 'finalization' if state['finalization_used']<state['limits'][1] else 'exhausted'


def finalize(attempt):
    """Freeze the existing draft, mark it unverified, let normal host evaluation judge it."""
    if attempt.budget()['submission']:return attempt.budget()['submission']
    # At the hard stop prefer a runnable, already evaluated package. The normal
    # select_parent operation retains the edited draft before restoring a parent.
    candidates=[]
    key=attempt.context['scope_id'].replace('/','-')
    pattern=f"{key}-development-r{attempt.context['round_index']:02d}-d*/evaluations/*/request.json"
    for path in attempt.store.path('meta/attempts').glob(pattern):
        record=json.loads(path.read_text())
        eid=f'dev-{path.parents[2].name}-{path.parent.name}'
        summary=attempt.pool.evaluation_status(eid)
        if summary.get('comparable'):
            candidates.append((path.stat().st_mtime,record['commit']))
    chosen=max(candidates)[1] if candidates else attempt.context['initial_base_commit']
    attempt.select_parent(chosen)
    return attempt.submit('Host finalization at the shared per-round Meta generation cap. '
        'Latest comparable internal candidate, or unchanged initial package if none. '
        'Draft retained separately; promotion still requires ordinary formal evaluation.',origin='round_limit')
