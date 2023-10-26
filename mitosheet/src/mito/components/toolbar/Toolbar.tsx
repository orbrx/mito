// Copyright (c) Mito

import React from 'react';
import "../../../../css/toolbar.css";
import { MitoAPI } from '../../api/api';
import { AnalysisData, EditorState, GridState, SheetData, UIState, UserProfile } from '../../types';
import { Actions } from '../../utils/actions';
import GetSupportButton from '../elements/GetSupportButton';
import PlanButton from './PlanButton';
import ToolbarCodeDropdown from './ToolbarCodeDropdown';
import ToolbarColumnsDropdown from './ToolbarColumnsDropdown';
import ToolbarDataframesDropdown from './ToolbarDataframesDropdown';
import ToolbarMenu from './ToolbarDropdownSelector';
import ToolbarEditDropdown from './ToolbarEditDropdown';
import ToolbarFormatDropdown from './ToolbarFormatDropdown';
import ToolbarGraphsDropdown from './ToolbarGraphsDropdown';
import ToolbarHelpDropdown from './ToolbarHelpDropdown';
import ToolbarRowsDropdown from './ToolbarRowsDropdown.tsx';
import ToolbarViewDropdown from './ToolbarViewDropdown';
import ToolbarUserDefinedEditsDropdown from './ToolbarUserDefinedEditsDropdown';
import { HomeTabContents } from './HomeTabContents';
import { classNames } from '../../utils/classNames';

export const MITO_TOOLBAR_OPEN_SEARCH_ID = 'mito-open-search';
export const MITO_TOOLBAR_UNDO_ID = 'mito-undo-button';
export const MITO_TOOLBAR_REDO_ID = 'mito-redo-button';

export const Toolbar = (
    props: {
        mitoAPI: MitoAPI
        currStepIdx: number;
        lastStepIndex: number;
        highlightPivotTableButton: boolean;
        highlightAddColButton: boolean;
        actions: Actions;
        mitoContainerRef: React.RefObject<HTMLDivElement>;
        gridState: GridState;
        setGridState: React.Dispatch<React.SetStateAction<GridState>>;
        uiState: UIState;
        setUIState: React.Dispatch<React.SetStateAction<UIState>>;
        sheetData: SheetData;
        userProfile: UserProfile;
        setEditorState: React.Dispatch<React.SetStateAction<EditorState | undefined>>;
        analysisData: AnalysisData,
        sheetIndex: number,
        closeOpenEditingPopups: () => void
    }): JSX.Element => {  

    const [currentTab, setCurrentTab] = React.useState<string>('Home');
    const tabs: {[ tab: string ]: React.ReactElement } = {
        'Home': <HomeTabContents {...props}/>,
        'Test': <div> Testing </div>
    };

    const renderTabContents = () => {
        return tabs[currentTab] ?? <div> No tab found </div>;
    }

    return (
        <div className='mito-toolbar-container'>
            <div className='mito-toolbar-top'>
                <div className='mito-toolbar-top-left'>
                    <ToolbarMenu type='Edit' uiState={props.uiState} setUIState={props.setUIState}>
                        <ToolbarEditDropdown
                            actions={props.actions}
                            uiState={props.uiState}
                            setUIState={props.setUIState}
                            userProfile={props.userProfile}
                        />
                    </ToolbarMenu>
                    <ToolbarMenu type='Dataframes' uiState={props.uiState} setUIState={props.setUIState}>
                        <ToolbarDataframesDropdown
                            actions={props.actions}
                            uiState={props.uiState}
                            setUIState={props.setUIState}
                            userProfile={props.userProfile}
                        />
                    </ToolbarMenu>
                    <ToolbarMenu type='Columns' uiState={props.uiState} setUIState={props.setUIState}>
                        <ToolbarColumnsDropdown
                            actions={props.actions}
                            uiState={props.uiState}
                            setUIState={props.setUIState}
                            userProfile={props.userProfile}
                        />
                    </ToolbarMenu>
                    <ToolbarMenu type='Rows' uiState={props.uiState} setUIState={props.setUIState}>
                        <ToolbarRowsDropdown
                            actions={props.actions}
                            uiState={props.uiState}
                            setUIState={props.setUIState}
                            userProfile={props.userProfile}
                        />
                    </ToolbarMenu>
                    <ToolbarMenu type='Graphs' uiState={props.uiState} setUIState={props.setUIState}>
                        <ToolbarGraphsDropdown
                            actions={props.actions}
                            uiState={props.uiState}
                            setUIState={props.setUIState}
                            userProfile={props.userProfile}
                        />
                    </ToolbarMenu>
                    <ToolbarMenu type='Format' uiState={props.uiState} setUIState={props.setUIState}>
                        <ToolbarFormatDropdown
                            actions={props.actions}
                            uiState={props.uiState}
                            setUIState={props.setUIState}
                            userProfile={props.userProfile}
                        />
                    </ToolbarMenu>
                    <ToolbarMenu type='Code' uiState={props.uiState} setUIState={props.setUIState}>
                        <ToolbarCodeDropdown
                            actions={props.actions}
                            uiState={props.uiState}
                            setUIState={props.setUIState}
                            userProfile={props.userProfile}
                        />
                    </ToolbarMenu>
                    <ToolbarMenu type='View' uiState={props.uiState} setUIState={props.setUIState}>
                        <ToolbarViewDropdown
                            actions={props.actions}
                            uiState={props.uiState}
                            setUIState={props.setUIState}
                            userProfile={props.userProfile}
                        />
                    </ToolbarMenu>
                    {props.actions.runtimeEditActionsList.length > 0 &&
                        <ToolbarMenu type='Custom Edits' uiState={props.uiState} setUIState={props.setUIState}>
                            <ToolbarUserDefinedEditsDropdown
                                actions={props.actions}
                                uiState={props.uiState}
                                setUIState={props.setUIState}
                                userProfile={props.userProfile}
                            />
                        </ToolbarMenu>
                        
                    }
                    <ToolbarMenu type='Help' uiState={props.uiState} setUIState={props.setUIState}>
                        <ToolbarHelpDropdown
                            actions={props.actions}
                            uiState={props.uiState}
                            setUIState={props.setUIState}
                            userProfile={props.userProfile}
                        />
                    </ToolbarMenu>
                </div>
                <div className='mito-toolbar-top-right'>
                    <GetSupportButton 
                        userProfile={props.userProfile} 
                        setUIState={props.setUIState} 
                        mitoAPI={props.mitoAPI} 
                        width='hug-contents'
                        className='mito-plan-button'
                    />
                    <PlanButton
                        uiState={props.uiState}
                        userProfile={props.userProfile}
                        setUIState={props.setUIState}
                        mitoAPI={props.mitoAPI}
                    />
                </div>
            </div>
            <div className='mito-toolbar-top-bottom-seperator'/>
            <div className='mito-toolbar-tabbar'>
                {Object.keys(tabs).map((tab) => {
                    return <button
                        key={tab}
                        onClick={() => setCurrentTab(tab)}
                        className={classNames('mito-toolbar-tabbar-button', currentTab === tab ? 'mito-toolbar-tabbar-button-selected' : '')}
                    >
                        <span>{tab}</span>
                        {currentTab === tab && <div className='mito-toolbar-tabbar-selected-underline'/>}
                    </button>
                })}
            </div>
            {renderTabContents()}
        </div>
    );
};
