"""Load-induced health probe failures must not spawn duplicate model services."""
from types import SimpleNamespace
from meta_harness.services import ServiceManager


def test_busy_borrowed_owner_is_not_restarted(tmp_path,monkeypatch):
    m=ServiceManager(tmp_path,SimpleNamespace(evaluation_retries=3))
    owner={'pid':123}
    m._write('services/vla-0/state.json',{'state':'ready','managed':False,'owner':owner,'ready_at':1,'endpoint':'http://localhost:8620'})
    monkeypatch.setattr('meta_harness.services.owns_process',lambda _:True)
    class Busy:
        def __init__(self,*args):pass
        def call(self,*args,**kwargs):raise TimeoutError('busy')
    monkeypatch.setattr('meta_harness.services.HttpRpcClient',Busy)
    result=m._tick_one('vla-0',{'endpoint':'http://localhost:8620','session_protocol':'pi05'})
    assert result['state']=='ready' and result['managed'] is False
    assert result['owner']==owner and result['health_warning']


def test_existing_healthy_listener_recovers_stale_failed_launch(tmp_path,monkeypatch):
    m=ServiceManager(tmp_path,SimpleNamespace(evaluation_retries=3))
    m._write('services/vla-0/state.json',{'state':'failed','managed':True,'attempt':4})
    m._write('services/vla-0/owner.json',{'pid':123})
    monkeypatch.setattr('meta_harness.services.owns_process',lambda _:False)
    monkeypatch.setattr('meta_harness.services.local_service_owner',lambda _:{'pid':456})
    class Healthy:
        def __init__(self,*args):pass
        def call(self,method,**kwargs):return {'protocol':'pi05','version':1} if method=='vla.protocol' else True
    monkeypatch.setattr('meta_harness.services.HttpRpcClient',Healthy)
    result=m._tick_one('vla-0',{'endpoint':'http://localhost:8620','session_protocol':'pi05'})
    assert result['state']=='ready' and result['owner']=={'pid':456}
    assert result['managed'] is False and result['attempt']==0
