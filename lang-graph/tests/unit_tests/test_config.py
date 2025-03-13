import os
import json
import pytest
from unittest.mock import patch, mock_open

from political_agent_graph.config import (
    select_persona,
    get_active_persona,
    get_model_for_task,
    get_temperature_for_task,
    DEFAULT_PERSONA,
    PERSONA_MODEL_MAP,
    CONFIG_FILE,
    TASK_CONFIG,
)


@pytest.fixture
def mock_config_file_exists():
    """Fixture to mock a config file that exists"""
    mock_config = {
        "active_persona": "trump"
    }
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=json.dumps(mock_config))):
        yield mock_config


@pytest.fixture
def mock_config_file_not_exists():
    """Fixture to mock a config file that doesn't exist"""
    with patch("os.path.exists", return_value=False):
        yield


class TestConfig:
    def test_get_active_persona_with_existing_config(self, mock_config_file_exists):
        """Test get_active_persona when the config file exists"""
        persona = get_active_persona()
        assert persona == "trump"

    def test_get_active_persona_with_no_config(self, mock_config_file_not_exists):
        """Test get_active_persona when no config file exists"""
        with patch("builtins.open", mock_open(read_data="{}")):
            persona = get_active_persona()
            assert persona == DEFAULT_PERSONA

    def test_select_persona_creates_config(self, mock_config_file_not_exists):
        """Test select_persona creates a config file with the right persona"""
        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            select_persona("biden")
            
            # Check the file was opened for writing
            mock_file.assert_called_once_with(CONFIG_FILE, "w")
            
            # Check the correct content was written
            write_call = mock_file().write.call_args[0][0]
            assert json.loads(write_call)["active_persona"] == "biden"

    def test_select_persona_updates_existing_config(self, mock_config_file_exists):
        """Test select_persona updates an existing config file"""
        mock_file = mock_open(read_data=json.dumps({"active_persona": "trump"}))
        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", mock_file):
            select_persona("biden")
            
            # Check the file was opened for reading and writing
            assert mock_file.call_count >= 2
            
            # Check the correct content was written
            write_call = mock_file().write.call_args[0][0]
            assert json.loads(write_call)["active_persona"] == "biden"

    def test_select_invalid_persona(self):
        """Test selecting an invalid persona raises ValueError"""
        with pytest.raises(ValueError):
            select_persona("invalid_persona")

    def test_get_model_for_task(self):
        """Test get_model_for_task returns correct model for task and persona"""
        for persona in PERSONA_MODEL_MAP.keys():
            for task in TASK_CONFIG.keys():
                with patch("political_agent_graph.config.get_active_persona", return_value=persona):
                    model = get_model_for_task(task)
                    expected_model = PERSONA_MODEL_MAP[persona].get(task, PERSONA_MODEL_MAP[persona].get("default"))
                    assert model == expected_model

    def test_get_model_for_invalid_task(self):
        """Test get_model_for_task with invalid task returns default model"""
        with patch("political_agent_graph.config.get_active_persona", return_value="trump"):
            model = get_model_for_task("invalid_task")
            assert model == PERSONA_MODEL_MAP["trump"]["default"]

    def test_get_temperature_for_task(self):
        """Test get_temperature_for_task returns correct temperature for task"""
        for task in TASK_CONFIG.keys():
            temp = get_temperature_for_task(task)
            expected_temp = TASK_CONFIG[task].get("temperature", 0.7)  # 0.7 is default from the code
            assert temp == expected_temp

    def test_get_temperature_for_invalid_task(self):
        """Test get_temperature_for_task with invalid task returns default temperature"""
        temp = get_temperature_for_task("invalid_task")
        assert temp == 0.7  # Default temperature

