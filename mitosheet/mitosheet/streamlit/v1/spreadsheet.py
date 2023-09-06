from collections import OrderedDict
import hashlib
import inspect
import json
import os
import pickle
from typing import Any, Dict, List, Callable, Optional, Tuple, Union

import pandas as pd

from mitosheet.mito_backend import MitoBackend
from mitosheet.types import CodeOptions
from mitosheet.utils import get_new_id

def _get_dataframe_hash(df: pd.DataFrame) -> bytes:
    """
    Returns a hash for a pandas dataframe that is consistent across runs, notably including:
    1. The column names
    2. The values of the dataframe
    3. The index of the dataframe
    4. The order of all of these

    This is necessary due to the issues described here: https://github.com/streamlit/streamlit/issues/7086
    where streamlit default hashing is not ideal for pandas dataframes, as it misses some column header and
    reordering changes. 
    """
    try:
        return hashlib.md5(
            bytes(str(pd.util.hash_pandas_object(df.columns)), 'utf-8') +
            bytes(str(pd.util.hash_pandas_object(df)), 'utf-8')
        ).digest()
    except TypeError as e:        
        # Use pickle if pandas cannot hash the object for example if
        # it contains unhashable objects.
        return b"%s" % pickle.dumps(df, pickle.HIGHEST_PROTOCOL)

def get_dataframe_hash(df: pd.DataFrame) -> bytes:
    _PANDAS_ROWS_LARGE = 100000
    _PANDAS_SAMPLE_SIZE = 10000
    
    if len(df) >= _PANDAS_ROWS_LARGE:
        df = df.sample(n=_PANDAS_SAMPLE_SIZE, random_state=0)
    
    return _get_dataframe_hash(df)


def get_function_from_code_unsafe(code: str) -> Optional[Callable]:
    """
    Given a string of code, returns the first function defined in the code. Notably, to do
    this, it executes the code, and then returns the first function defined in the code. 

    As it executes the full code string, you should only use this function if you trust the
    code string -- and in our case, if the function is not called.

    If no functions are defined, returns None
    """
    functions_before = [f for f in locals().values() if callable(f)]
    exec(code)
    functions = [f for f in locals().values() if callable(f) and f not in functions_before]
    new_functions = []
    # We then find the one function that was defined inside of this module -- as the above 
    # exec likely defines all the other mitosheet functions (none of which we actaully want)
    for f in functions:
        if inspect.getmodule(f) == inspect.getmodule(get_function_from_code_unsafe):
            new_functions.append(f)

    if len(new_functions) == 0:
        return None

    return new_functions[0]



def get_selected_element(dfs: List[pd.DataFrame], indexAndSelections: Any) -> Union[pd.DataFrame, pd.Series, None]:

    if indexAndSelections is None:
        return None

    selected_dataframe_index = indexAndSelections['selectedDataframeIndex']
    if selected_dataframe_index < 0 or selected_dataframe_index >= len(dfs):
        return None
    
    df = dfs[selected_dataframe_index]

    # If there are multiple selections, for now we only return the first one - for simplicity in return types
    selection = next(iter(indexAndSelections['selections']))

    # Selections have the format: {'startingRowIndex': 0, 'endingRowIndex': 0, 'startingColumnIndex': 5, 'endingColumnIndex': 5}

    smallerRowIndex = min(selection['startingRowIndex'], selection['endingRowIndex'])
    largerRowIndex = max(selection['startingRowIndex'], selection['endingRowIndex'])
    smallerColumnIndex = min(selection['startingColumnIndex'], selection['endingColumnIndex'])
    largerColumnIndex = max(selection['startingColumnIndex'], selection['endingColumnIndex'])

    # If the row indexes selected are both -1, we just return the column
    if smallerRowIndex == -1 and largerRowIndex == -1:
        return df.iloc[:, smallerColumnIndex:largerColumnIndex + 1]
    
    # If the column indexes selected are both -1, we just return the row
    if smallerColumnIndex == -1 and largerColumnIndex == -1:
        return df.iloc[smallerRowIndex:largerRowIndex + 1, :]
    
    # If one row index is -1, then we return the column
    if smallerRowIndex == -1:
        return df.iloc[:, smallerColumnIndex:largerColumnIndex + 1]
    
    # If one column index is -1, then we return the row
    if smallerColumnIndex == -1:
        return df.iloc[smallerRowIndex:largerRowIndex + 1, :]
    
    # Otherwise, we return the intersection of the row and column
    return df.iloc[smallerRowIndex:largerRowIndex + 1, smallerColumnIndex:largerColumnIndex + 1]
    

try:
    import streamlit.components.v1 as components
    import streamlit as st

    parent_dir = os.path.dirname(os.path.abspath(__file__))

    mito_build_dir = os.path.join(parent_dir, "mitoBuild")
    _mito_component_func = components.declare_component("my_component", path=mito_build_dir)

    message_passer_build_dr = os.path.join(parent_dir, "messagingBuild")
    _message_passer_component_func = components.declare_component("message-passer", path=message_passer_build_dr)


    def get_session_id() -> Optional[str]:
        """
        This returns the session id for the current script run. Notably, this is different
        when the user:
        1. Refreshes the page
        2. Closes the page and reopens it
        3. Is a different user
        4. Is a different browser

        It allows us to cache the Mito backend on the session id, so that it is not
        shared across users. Notably, it clearing when refreshed is the same behavior
        that streamlit caching has by default -- e.g. the button component will reset
        when the page is refreshed.

        From the streamlit docs:
        A context object that contains data for a "script run" - that is,
        data that's scoped to a single ScriptRunner execution (and therefore also
        scoped to a single connected "session").
        """
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        if ctx is None:
            return None
        return ctx.session_id

    @st.cache_resource(hash_funcs={pd.DataFrame: get_dataframe_hash})
    def _get_mito_backend(
            *args: Union[pd.DataFrame, str, None], 
            _importers: Optional[List[Callable]]=None, 
            _sheet_functions: Optional[List[Callable]]=None, 
            _code_options: Optional[CodeOptions]=None,
            import_folder: Optional[str]=None,
            df_names: Optional[List[str]]=None,
            session_id: Optional[str]=None,
            analysis_name: Optional[str]=None,
            key: Optional[str]=None # So it caches on key
        ) -> Tuple[MitoBackend, List[Any]]: 

        mito_backend = MitoBackend(
            *args, 
            import_folder=import_folder,
            user_defined_importers=_importers, user_defined_functions=_sheet_functions,
            code_options=_code_options,
            analysis_to_replay=analysis_name,
        )

        # Make a send function that stores the responses in a list
        responses = []
        def send(response):
            responses.append(response)
        
        mito_backend.mito_send = send

        if df_names is not None and len(df_names) > 0:
            mito_backend.receive_message(
                {
                    'event': 'update_event',
                    'id': get_new_id(),
                    'type': 'args_update',
                    'params': {
                        'args': df_names
                    },
                }
            )

        return mito_backend, responses

    def message_passer_component(key: Optional[str]=None) -> Any:
        """
        This component simply passes messages from the frontend to the backend,
        so that the backend can process them before it is rendered.
        """
        component_value = _message_passer_component_func(key=key)
        return component_value


    def spreadsheet( # type: ignore
            *args: Union[pd.DataFrame, str, None], 
            sheet_functions: Optional[List[Callable]]=None, 
            importers: Optional[List[Callable]]=None, 
            df_names: Optional[List[str]]=None,
            import_folder: Optional[str]=None,
            code_options: Optional[CodeOptions]=None,
            return_type: str='default',
            analysis_name: Optional[str]=None,
            key=None
        ) -> Any:
        """
        Create a new instance of the Mito spreadsheet in a streamlit app.

        Parameters
        ----------
        args: pd.Dataframe or str or None
            The arguments to pass to the Mito spreadsheet. If a dataframe is
            passed, it will be displayed as a sheet tab. If a string is passed,
            it will be read in with a pd.read_csv call. If None is passed, it 
            will be skipped.
        sheet_functions: List[Callable]
            A list of functions that can be used in the spreadsheet. Functions
            should be capitalized.
        importers: List[Callable]
            A list of functions that can be used to import dataframes. Each
            function should return a dataframe. 
        df_names: List[str]
            A list of names for the dataframes passed in. If None, the dataframes
            will be named df0, df1, etc.
        key: str or None
            An key that uniquely identifies this component. This must be passed
            for now, or the component will not work. Not sure why.

        Returns
        -------
        Tuple[Dict[str, pd.DataFrame], List[str]]
            A tuple. The first element is a mapping from dataframe names to the
            final dataframes. The second element is a list of lines of code
            that were executed in the Mito spreadsheet.
        """
        # Get the absolute path to the import_folder, in case it is relative. Also
        # check that this folder exists, and throw an error if it does not.
        if import_folder is not None:
            import_folder = os.path.expanduser(import_folder)
            import_folder = os.path.abspath(import_folder)

            if not os.path.exists(import_folder):
                raise ValueError(f"Import folder {import_folder} does not exist. Please change the file path or create the folder.")

        session_id = get_session_id()

        mito_backend, responses = _get_mito_backend(
            *args, 
            _sheet_functions=sheet_functions,
            _importers=importers, 
            _code_options=code_options,
            import_folder=import_folder,
            session_id=session_id,
            df_names=df_names, 
            analysis_name=analysis_name,
            key=key
        )

        # Mito widgets need new ids every time a new one is displayed. As such, if
        # the key is None, we generate a new one. Notably, we do this after getting the
        # mito_backend, so that we can cache the mito_backend on the user provided key.
        if key is None:
            key = mito_backend.analysis_name

        sheet_data_json = mito_backend.steps_manager.sheet_data_json,
        analysis_data_json = mito_backend.steps_manager.analysis_data_json,
        user_profile_json = mito_backend.get_user_profile_json()

        msg = message_passer_component(key=str(key) + 'message_passer')
        if (
            msg is not None \
            and msg['id'] not in [response['id'] for response in responses] \
            and msg['analysis_name'] == mito_backend.analysis_name
        ):
            # We receive a message if:
            # 1. It is not None
            # 2. We have not already received it on this backend
            # 3. It is for this analysis. 
            # Note that the final two conditions are to prevent messages that have been sent
            # by the message passer component from being received again. This happens because
            # when a component value is set, it is always returned by the message_passer_component
            # until a new component value is set            
            mito_backend.receive_message(msg)
            
        responses_json = json.dumps(responses)

        # NOTE: selection is Optional -- as if the user has not set the return type as selected, we don't
        # waste a component value update setting the value
        selection = _mito_component_func(
            key=key, 
            sheet_data_json=sheet_data_json, analysis_data_json=analysis_data_json, user_profile_json=user_profile_json, 
            responses_json=responses_json, id=id(mito_backend),
            return_type=return_type
        )

        # We return a mapping from dataframe names to dataframes
        final_state = mito_backend.steps_manager.curr_step.final_defined_state
        code = "\n".join(mito_backend.steps_manager.code())

        ordered_dict = OrderedDict()
        for df_name, df in zip(final_state.df_names, final_state.dfs):
            ordered_dict[df_name] = df

        if return_type == 'default':
            return ordered_dict, code
        elif return_type == 'selection':
            return get_selected_element(final_state.dfs, selection)
        elif return_type == 'default_list':
            return final_state.dfs, code
        elif return_type == 'dfs':
            return ordered_dict
        elif return_type == 'code':
            return code
        elif return_type == 'dfs_list':
            return final_state.dfs
        elif return_type == 'function':
            if code_options is None or not code_options['as_function'] or code_options['call_function']:
                raise ValueError(f"""You must set code_options with `as_function=True` and `call_function=False` in order to return a function.""")
            
            return get_function_from_code_unsafe(code)
        else:
            raise ValueError(f'Invalid value for return_type={return_type}. Must be "default", "default_list", "dfs", "code", "dfs_list", or "function".')

    
except ImportError:
    def spreadsheet(*args, key=None): # type: ignore
        raise RuntimeError("Couldn't import streamlit. Install streamlit with `pip install streamlit` to use the mitosheet component.")