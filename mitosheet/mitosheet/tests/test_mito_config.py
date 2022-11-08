#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Mito.
# Distributed under the terms of the Modified BSD License.

from mitosheet.mito_config import MITO_CONFIG_SUPPORT_EMAIL_KEY, MITO_CONFIG_VERSION_KEY, MitoConfig
import os

def delete_env_var_if_exists(env_var: str) -> None: 
    """
    Deletes the environment variable only if it exists to avoid errors. Helpful for testing.
    """
    if os.environ.get(env_var) is not None:
        del os.environ[env_var]

def test_keys_did_not_change():
    assert MITO_CONFIG_VERSION_KEY == 'MITO_CONFIG_VERSION'
    assert MITO_CONFIG_SUPPORT_EMAIL_KEY == 'MITO_CONFIG_SUPPORT_EMAIL'

def test_none_works():
    # Delete the environmnet variables so we can test the none condition
    delete_env_var_if_exists(MITO_CONFIG_SUPPORT_EMAIL_KEY)
    delete_env_var_if_exists(MITO_CONFIG_VERSION_KEY)

    mito_config = MitoConfig()
    mito_config_dict = mito_config.get_mito_config()
    assert mito_config_dict == {
        MITO_CONFIG_VERSION_KEY: '1',
        MITO_CONFIG_SUPPORT_EMAIL_KEY: 'founders@sagacollab.com'
    }


def test_none_config_version_key_is_string():
    # Delete the environmnet variables so we can test the none condition
    delete_env_var_if_exists(MITO_CONFIG_SUPPORT_EMAIL_KEY)
    delete_env_var_if_exists(MITO_CONFIG_VERSION_KEY)

    mito_config = MitoConfig()
    mito_config_dict = mito_config.get_mito_config()
    assert isinstance(mito_config_dict[MITO_CONFIG_VERSION_KEY], str)

def test_version_1_works():
    # Set environment variables
    os.environ[MITO_CONFIG_VERSION_KEY] = "1"
    os.environ[MITO_CONFIG_SUPPORT_EMAIL_KEY] = "aaron@sagacollab.com"
    
    # Test reading environment variables works properly
    mito_config = MitoConfig()
    assert mito_config.get_mito_config() == {
        MITO_CONFIG_VERSION_KEY: '1',
        MITO_CONFIG_SUPPORT_EMAIL_KEY: 'aaron@sagacollab.com'
    }    

    # Delete the environmnet variables for the next test
    delete_env_var_if_exists(MITO_CONFIG_SUPPORT_EMAIL_KEY)
    delete_env_var_if_exists(MITO_CONFIG_VERSION_KEY)


