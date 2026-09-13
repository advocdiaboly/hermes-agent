"""Own-profile policy must gate every DB-opening path, not just discovery."""
import asyncio
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_state import SessionDB
from tools.session_search_tool import session_search


@contextmanager
def scoped(home):
    token = set_hermes_home_override(str(home))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


@pytest.fixture
def homes(tmp_path, monkeypatch):
    root = tmp_path / '.hermes'
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    result = {}
    for name, restricted in [('dima', True), ('family', False)]:
        home = root / 'profiles' / name
        home.mkdir(parents=True)
        (home / 'config.yaml').write_text(
            'session_search:\n  current_profile_only: ' + str(restricted).lower() + '\n')
        db = SessionDB(home / 'state.db')
        db.create_session(name, source='telegram')
        db.append_message(name, role='user', content=f'{name} confidential canary')
        anchor = db._conn.execute('select id from messages limit 1').fetchone()[0]
        result[name] = home, db, anchor
    yield result
    for _, db, _ in result.values():
        db.close()


def test_restriction_covers_all_read_shapes_and_preserves_own_history(homes):
    home, db, anchor = homes['dima']
    foreign_anchor = homes['family'][2]
    with scoped(home):
        for args in [dict(profile='family'), dict(profile='family', query='canary'),
                     dict(profile='family', session_id='family'),
                     dict(profile='family', session_id='family', around_message_id=foreign_anchor),
                     dict(session_id='family/family'),
                     dict(session_id='@session:family/family'), dict(session_id='family')]:
            result = json.loads(session_search(db=db, **args))
            assert result['success'] is False, args
            assert 'family confidential canary' not in json.dumps(result)
        for args in [{}, dict(query='canary'), dict(profile='dima', session_id='dima'),
                     dict(session_id='dima/dima'), dict(session_id='@session:dima/dima'),
                     dict(session_id='dima', around_message_id=anchor)]:
            result = json.loads(session_search(db=db, **args))
            assert result['success'] is True, result
            assert 'dima confidential canary' in json.dumps(result)


def test_policy_is_context_local_and_invalid_policy_never_grants_access(homes):
    async def read(name, target):
        home, db, _ = homes[name]
        with scoped(home):
            await asyncio.sleep(0)
            return json.loads(session_search(db=db, profile=target, session_id=target))

    async def interleaved():
        return await asyncio.gather(read('dima', 'family'), read('family', 'dima'))

    blocked, allowed = asyncio.run(interleaved())
    assert blocked['success'] is False
    assert allowed['success'] is True
    home, db, _ = homes['family']
    for invalid in ['session_search:\n  current_profile_only: "false"\n',
                    'session_search: invalid\n', 'session_search: [\n']:
        (home / 'config.yaml').write_text(invalid)
        with scoped(home):
            result = json.loads(session_search(db=db, profile='dima', session_id='dima'))
            assert result['success'] is False
            assert 'dima confidential canary' not in json.dumps(result)
