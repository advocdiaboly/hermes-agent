"""Tests for docker container_config key propagation in file_tools."""

from unittest.mock import patch, MagicMock
import tools.file_tools as file_tools
import tools.code_execution_tool as code_execution_tool
import tools.terminal_tool as terminal_tool


def _make_env_config(**overrides):
    base = {
        "env_type": "docker",
        "docker_image": "test-image:latest",
        "singularity_image": "docker://test",
        "modal_image": "test",
        "daytona_image": "test",
        "cwd": "/workspace",
        "host_cwd": None,
        "timeout": 180,
        "container_cpu": 2,
        "container_memory": 4096,
        "container_disk": 20480,
        "container_persistent": False,
        "modal_mode": "direct",
        "docker_volumes": [],
        "docker_mount_cwd_to_workspace": True,
        "docker_forward_env": ["MY_SECRET", "API_KEY"],
        "docker_env": {"LANG": "C.UTF-8"},
        "docker_run_as_host_user": True,
        "docker_extra_args": ["--user", "1000:1000", "--group-add", "1001"],
        "docker_network": False,
        "docker_persist_across_processes": False,
        "docker_orphan_reaper": False,
    }
    base.update(overrides)
    return base


class TestFileToolsContainerConfig:
    def _run(self, env_config, task_id, task_env_overrides=None):
        captured = {}
        mock_env = MagicMock()

        def fake_create_env(**kwargs):
            captured.update(kwargs)
            return mock_env

        with patch("tools.terminal_tool._get_env_config", return_value=env_config), \
             patch("tools.terminal_tool._task_env_overrides", task_env_overrides or {}), \
             patch("tools.terminal_tool._active_environments", {}), \
             patch("tools.terminal_tool._creation_locks", {}), \
             patch("tools.terminal_tool._creation_locks_lock", __import__("threading").Lock()), \
             patch("tools.terminal_tool._create_environment", side_effect=fake_create_env), \
             patch("tools.terminal_tool._start_cleanup_thread"), \
             patch("tools.terminal_tool._check_disk_usage_warning"), \
             patch("tools.file_tools._file_ops_cache", {}), \
             patch("tools.file_tools._file_ops_lock", __import__("threading").Lock()):
            file_tools._get_file_ops(task_id)

        return captured

    def test_docker_mount_cwd_to_workspace_passed(self):
        """docker_mount_cwd_to_workspace is forwarded to container_config."""
        cc = self._run(_make_env_config(docker_mount_cwd_to_workspace=True), "t1").get("container_config", {})
        assert cc.get("docker_mount_cwd_to_workspace") is True

    def test_docker_forward_env_passed(self):
        """docker_forward_env is forwarded to container_config."""
        cc = self._run(_make_env_config(docker_forward_env=["MY_SECRET"]), "t2").get("container_config", {})
        assert cc.get("docker_forward_env") == ["MY_SECRET"]

    def test_docker_mount_cwd_defaults_to_false(self):
        """docker_mount_cwd_to_workspace defaults to False when absent from config."""
        cfg = _make_env_config()
        del cfg["docker_mount_cwd_to_workspace"]
        cc = self._run(cfg, "t3").get("container_config", {})
        assert cc.get("docker_mount_cwd_to_workspace") is False

    def test_docker_forward_env_defaults_to_empty_list(self):
        """docker_forward_env defaults to [] when absent from config."""
        cfg = _make_env_config()
        del cfg["docker_forward_env"]
        cc = self._run(cfg, "t4").get("container_config", {})
        assert cc.get("docker_forward_env") == []

    def test_all_container_contract_keys_are_passed(self):
        """File tools preserve the complete terminal container contract."""
        config = _make_env_config()
        captured = self._run(config, "file-contract")
        assert captured["container_config"] == {
            key: config[key]
            for key in (
                "container_cpu",
                "container_memory",
                "container_disk",
                "container_persistent",
                "modal_mode",
                "docker_volumes",
                "docker_mount_cwd_to_workspace",
                "docker_forward_env",
                "docker_env",
                "docker_run_as_host_user",
                "docker_extra_args",
                "docker_network",
                "docker_persist_across_processes",
                "docker_orphan_reaper",
            )
        }

    def test_cwd_only_raw_task_override_reaches_file_environment(self):
        """CWD-only task overrides collapse to default but must keep their cwd."""
        captured = self._run(
            _make_env_config(env_type="local", cwd="/config-cwd"),
            "desktop-session-cwd",
            task_env_overrides={"desktop-session-cwd": {"cwd": "/workspace/session"}},
        )

        assert captured["task_id"] == "default"
        assert captured["cwd"] == "/workspace/session"


def test_code_execution_passes_the_complete_container_contract():
    """Execute-code receives the same Docker options as terminal and file tools."""
    config = _make_env_config()
    captured = {}
    mock_env = MagicMock()

    def fake_create_env(**kwargs):
        captured.update(kwargs)
        return mock_env

    with patch("tools.terminal_tool._get_env_config", return_value=config), \
         patch("tools.terminal_tool._task_env_overrides", {}), \
         patch("tools.terminal_tool._active_environments", {}), \
         patch("tools.terminal_tool._creation_locks", {}), \
         patch("tools.terminal_tool._creation_locks_lock", __import__("threading").Lock()), \
         patch("tools.terminal_tool._resolve_container_task_id", side_effect=lambda task_id: task_id), \
         patch("tools.terminal_tool._create_environment", side_effect=fake_create_env), \
         patch("tools.terminal_tool._start_cleanup_thread"):
        code_execution_tool._get_or_create_env("code-contract")

    assert captured["container_config"] == {
        key: config[key]
        for key in (
            "container_cpu",
            "container_memory",
            "container_disk",
            "container_persistent",
            "modal_mode",
            "docker_volumes",
            "docker_mount_cwd_to_workspace",
            "docker_forward_env",
            "docker_env",
            "docker_run_as_host_user",
            "docker_extra_args",
            "docker_network",
            "docker_persist_across_processes",
            "docker_orphan_reaper",
        )
    }


def test_shared_container_config_keeps_terminal_defaults():
    """The shared builder preserves terminal's established default contract."""
    assert terminal_tool._build_container_config({}) == {
        "container_cpu": 1,
        "container_memory": 5120,
        "container_disk": 51200,
        "container_persistent": True,
        "modal_mode": "auto",
        "docker_volumes": [],
        "docker_mount_cwd_to_workspace": False,
        "docker_forward_env": [],
        "docker_env": {},
        "docker_run_as_host_user": False,
        "docker_extra_args": [],
        "docker_network": True,
        "docker_persist_across_processes": True,
        "docker_orphan_reaper": True,
    }


def test_terminal_creation_path_passes_shared_container_contract():
    """Terminal creation passes the shared contract to _create_environment."""
    config = _make_env_config()
    captured = {}
    mock_env = MagicMock(cwd=config["cwd"])
    mock_env.execute.return_value = {"output": "", "returncode": 0}

    def fake_create_env(**kwargs):
        captured.update(kwargs)
        return mock_env

    with patch("tools.terminal_tool._get_env_config", return_value=config), \
         patch("tools.terminal_tool._task_env_overrides", {}), \
         patch("tools.terminal_tool._active_environments", {}), \
         patch("tools.terminal_tool._creation_locks", {}), \
         patch("tools.terminal_tool._creation_locks_lock", __import__("threading").Lock()), \
         patch("tools.terminal_tool._resolve_container_task_id", side_effect=lambda task_id: task_id), \
         patch("tools.terminal_tool.resolve_task_overrides", return_value={}), \
         patch("tools.terminal_tool.get_session_cwd", return_value=None), \
         patch("tools.terminal_tool.record_session_cwd"), \
         patch("tools.terminal_tool._create_environment", side_effect=fake_create_env), \
         patch("tools.terminal_tool._build_container_config",
               wraps=terminal_tool._build_container_config) as build_container_config, \
         patch("tools.terminal_tool._start_cleanup_thread"), \
         patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
        terminal_tool.terminal_tool("true", task_id="terminal-contract")

    build_container_config.assert_called_once_with(config)
    assert captured["container_config"] == terminal_tool._build_container_config(config)
    assert captured["container_config"]["docker_extra_args"] == [
        "--user",
        "1000:1000",
        "--group-add",
        "1001",
    ]
