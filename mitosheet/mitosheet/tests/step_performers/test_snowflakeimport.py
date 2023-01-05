#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Saga Inc.
# Distributed under the terms of the GPL License.
"""
Contains tests for SnowflakeImport
"""

import pandas as pd
import pytest
from mitosheet.tests.test_utils import create_mito_wrapper_dfs

SNOWFLAKEIMPORT_TESTS = [
    (
        [
            pd.DataFrame({'A': [1, 2, 3], 'B': [1.0, 2.0, 3.0], 'C': [True, False, True], 'D': ["string", "with spaces", "and/!other@characters"], 'E': pd.to_datetime(['12-22-1997', '12-23-1997', '12-24-1997']), 'F': pd.to_timedelta(['1 days', '2 days', '3 days'])})
        ],
        "put", 
        "params", 
        "here",
        [
            pd.DataFrame({'A': [1, 2, 3], 'B': [1.0, 2.0, 3.0], 'C': [True, False, True], 'D': ["string", "with spaces", "and/!other@characters"], 'E': pd.to_datetime(['12-22-1997', '12-23-1997', '12-24-1997']), 'F': pd.to_timedelta(['1 days', '2 days', '3 days'])})
        ]
    ),
]
@pytest.mark.parametrize("input_dfs, snowflake_credentials, query_params, output_dfs", SNOWFLAKEIMPORT_TESTS)
def test_snowflakeimport(input_dfs, snowflake_credentials, query_params, output_dfs):
    mito = create_mito_wrapper_dfs(*input_dfs)

    mito.snowflakeimport(snowflake_credentials, query_params)

    assert len(mito.dfs) == len(output_dfs)
    for actual, expected in zip(mito.dfs, output_dfs):
        assert actual.equals(expected)