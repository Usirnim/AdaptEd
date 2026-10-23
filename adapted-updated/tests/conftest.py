import os, sys, tempfile, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "t.db")   # must be set before importing app
os.environ.pop("LLM_API_KEY", None)                                # tests never hit the network
import app as appmod, content

@pytest.fixture()
def client(tmp_path):
    appmod.DB = str(tmp_path / "t.db"); appmod.init(); appmod.app.config["TESTING"] = True
    return appmod.app.test_client()

@pytest.fixture()
def user(client):
    r = client.post("/api/auth/register", json={"name": "alice", "password": "secret"}); assert r.status_code == 200
    return client

ALL = {k: v for _, (_, cs) in content.SUBJECTS.items() for k, v in cs.items()}
CORPUS = [v[2] for v in ALL.values()]
