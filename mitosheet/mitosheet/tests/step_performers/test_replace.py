#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Saga Inc.
# Distributed under the terms of the GPL License.
"""
Contains tests for Replace
"""

import pandas as pd
import pytest
from mitosheet.tests.test_utils import create_mito_wrapper

from mitosheet.errors import MitoError

from mitosheet.utils import get_new_id

REPLACE_TESTS = [
    # Tests with boolean columns
    (
        [
            pd.DataFrame({
                'A': [1, 2, 3],
                'B': [1.0, 2.0, 3.0], 
                'C': [True, False, True], 
                'D': ["string", "with spaces", "and/!other@characters3"], 
                'E': pd.to_datetime(['12-22-1997', '12-23-1997', '12-24-1997']), 
                'F': pd.to_timedelta(['1 days', '2 days', '3 days'])
            })
        ],
        0,
        "3", 
        "4", 
        [
            pd.DataFrame({
                'A': [1, 2, 4],
                'B': [1.0, 2.0, 4.0], 
                'C': [True, False, True], 
                'D': ["string", "with spaces", "and/!other@characters4"], 
                'E': pd.to_datetime(['12-22-1997', '12-24-1997', '12-24-1997']), 
                'F': pd.to_timedelta(['1 days', '2 days', '4 days'])
            })
        ]
    ),(
        [
            pd.DataFrame({
                'A': [1, 2, 3],
                'B': [1.0, 2.0, 3.0], 
                'C': [True, False, True], 
                'D': ["string", "with spaces", "and/!other@characters3"], 
                'E': pd.to_datetime(['12-22-1997', '12-23-1997', '12-24-1997']), 
                'F': pd.to_timedelta(['1 days', '2 days', '3 days'])
            })
        ],
        0,
        "i", 
        "abc", 
        [
            pd.DataFrame({
                'A': [1, 2, 3],
                'B': [1.0, 2.0, 3.0], 
                'C': [True, False, True], 
                'D': ["strabcng", "wabcth spaces", "and/!other@characters3"], 
                'E': pd.to_datetime(['12-22-1997', '12-23-1997', '12-24-1997']), 
                'F': pd.to_timedelta(['1 days', '2 days', '3 days'])
            })
        ]
    ),(
        [
            pd.DataFrame({
                'A': [1, 2, 3],
                'B': [1.0, 2.0, 3.0], 
                'C': [True, False, True], 
                'D': ["string", "with spaces", "and/!other@characters3"], 
                'E': pd.to_datetime(['12-22-1997', '12-23-1997', '12-24-1997']), 
                'F': pd.to_timedelta(['1 days', '2 days', '3 days'])
            })
        ],
        0,
        "with spaces", 
        "abc", 
        [
            pd.DataFrame({
                'A': [1, 2, 3],
                'B': [1.0, 2.0, 3.0], 
                'C': [True, False, True], 
                'D': ["string", "abc", "and/!other@characters3"], 
                'E': pd.to_datetime(['12-22-1997', '12-23-1997', '12-24-1997']), 
                'F': pd.to_timedelta(['1 days', '2 days', '3 days'])
            })
        ]
    ),
    (
        [
            pd.DataFrame({
                'A': [1, 2, 3],
                'B': [1.0, 2.0, 3.0], 
                'C': [True, False, True], 
                'D': ["string", "with spaces", "and/!other@characters3"], 
                'E': pd.to_datetime(['12-22-1997', '12-23-1997', '12-24-1997']), 
                'F': pd.to_timedelta(['1 days', '2 days', '3 days'])
            })
        ],
        0,
        "true", 
        "false", 
        [
            pd.DataFrame({
                'A': [1, 2, 3],
                'B': [1.0, 2.0, 3.0], 
                'C': [False, False, False], 
                'D': ["string", "with spaces", "and/!other@characters3"], 
                'E': pd.to_datetime(['12-22-1997', '12-23-1997', '12-24-1997']), 
                'F': pd.to_timedelta(['1 days', '2 days', '3 days'])
            })
        ],
    ),

    # Tests without boolean columns
    (
        [
            pd.DataFrame({
                'A': [1, 2, 3],
                'B': [1.0, 2.0, 3.0], 
                'C': ["string", "with spaces", "and/!other@characters3"], 
                'D': pd.to_datetime(['12-22-1997', '12-23-1997', '12-24-1997']), 
            })
        ],
        0,
        "a", 
        "f", 
        [
            pd.DataFrame({
                'f': [1, 2, 3],
                'B': [1.0, 2.0, 3.0], 
                'C': ["string", "with spfces", "fnd/!other@chfrfcters3"], 
                'D': pd.to_datetime(['12-22-1997', '12-23-1997', '12-24-1997']), 
            })
        ],
    ),

    # Tests with strings that could interact with regex
    (
        [
            pd.DataFrame({
                'A': [1, 2, 3],
                'B': [1.0, 2.0, 3.0], 
                'C': ["'string.'", "with spaces", "and/!other@characters3"], 
                'D': pd.to_datetime(['12-22-1997', '12-23-1997', '12-24-1997']), 
            })
        ],
        0,
        "string.", 
        "f", 
        [
            pd.DataFrame({
                'A': [1, 2, 3],
                'B': [1.0, 2.0, 3.0], 
                'C': ["'f'", "with spaces", "and/!other@characters3"], 
                'D': pd.to_datetime(['12-22-1997', '12-23-1997', '12-24-1997']), 
            })
        ],
    ),
]
@pytest.mark.parametrize("input_dfs, sheet_index, search_value, replace_value, output_dfs", REPLACE_TESTS)
def test_replace(input_dfs, sheet_index, search_value, replace_value, output_dfs):
    mito = create_mito_wrapper(*input_dfs)

    mito.replace(sheet_index, search_value, replace_value)

    assert len(mito.dfs) == len(output_dfs)
    for actual, expected in zip(mito.dfs, output_dfs):
        print(actual)
        print(expected)
        pd.testing.assert_frame_equal(actual,expected)

REPLACE_INVALID_TESTS = [
    (
        [
            pd.DataFrame({
                'A': [1, 2, 3],
                'B': [1.0, 2.0, 3.0],
                'C': [True, False, True],
                'D': ["string", "with spaces", "and/!other@characters3"],
                'E': pd.to_datetime(['12-22-1997', '12-23-1997', '12-24-1997']),
                'F': pd.to_timedelta(['1 days', '2 days', '3 days'])
            })
        ],
        0,
        "3",
        "hi"
    ),
]

@pytest.mark.parametrize("input_dfs, sheet_index, search_value, replace_value", REPLACE_INVALID_TESTS)
def test_replace_invalid(input_dfs, sheet_index, search_value, replace_value):
    mito = create_mito_wrapper(*input_dfs)

    with pytest.raises(MitoError):
        mito.mito_backend.handle_edit_event(
            {
                'event': 'edit_event',
                'id': get_new_id(),
                'type': 'replace_edit',
                'step_id': get_new_id(),
                'params': {
                    'sheet_index': sheet_index,
                    'search_value': search_value,
                    'replace_value': replace_value,
                }
            }
        )