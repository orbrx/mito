import React from "react";
import { ChecklistID } from "./components/checklists/checklistData";
import { ModalInfo } from "./components/modals/modals";
import { ControlPanelTab } from "./components/taskpanes/ControlPanel/ControlPanelTaskpane";
import { GraphType } from "./components/taskpanes/Graph/GraphSetupTab";
import { TaskpaneInfo } from "./components/taskpanes/taskpanes";


/**
 * The different types of data manipulation that Mito supports.
 */
export enum StepType {
    Initialize = 'initialize',
    AddColumn = "add_column",
    DeleteColumn = "delete_column",
    RenameColumn = "rename_column",
    ReorderColumn = "reorder_column",
    FillNa = 'fill_na',
    FilterColumn = 'filter_column',
    SetColumnFormula = "set_column_formula",
    DataframeDelete = 'dataframe_delete',
    DataframeDuplicate = 'dataframe_duplicate',
    DataframeRename = 'dataframe_rename',
    SimpleImport = 'simple_import',
    Sort = 'sort',
    Pivot = 'pivot',
    Merge = 'merge',
    Concat = 'concat',
    DropDuplicates = 'drop_duplicates',
    ChangeColumnDtype = 'change_column_dtype',
    SetCellValue = 'set_cell_value',
    BulkOldRename = 'bulk_old_rename',
    ExcelImport = 'excel_import',
    Graph = 'graph',
    GraphDuplicate = 'graph_duplicate',
    GraphDelete = 'graph_delete',
    GraphRename = 'graph_rename',
    DeleteRow = 'delete_row',
    PromoteRowToHeader = 'promote_row_to_header',
    SplitTextToColumns = 'split_text_to_columns',
    Transpose = 'transpose',
    Melt = 'melt',
    OneHotEncoding = 'one_hot_encoding',
    SetDataframeFormat = 'set_dataframe_format',
    DataframeImport = 'dataframe_import',
    // AUTOGENERATED LINE: STEPTYPE (DO NOT DELETE)
}

export enum UpdateType {
    Undo = 'undo',
    Redo = 'redo',
    Clear = 'clear',
    ArgsUpdate = 'args_update',
    SaveAnalysisUpdate = 'save_analysis_update',
    ReplayAnalysisUpdate = 'replay_analysis_update',
    CheckoutStepByIdxUpdate = 'checkout_step_by_idx_update',
    AppendUserFieldUpdate = 'append_user_field_update',
    SetUserFieldUpdate = 'set_user_field_update',
    UpdateFeedbackv2ObjObject = 'update_feedback_v2_obj_update',
    GoPro = 'go_pro',
    RenderCountUpdate = 'render_count_update',
    ChecklistUpdate = 'checklist_update',
    UpdateExistingImportUpdate = 'update_existing_import_update',
}

/**
 * The summary of a single step that has occured.
 * 
 * @param step_id - the unique ID of this specific step
 * @param step_idx - the index of the step among the currently run steps
 * @param step_type - the type of this step
 * @param step_display_name - the highest level name to describe what happened in this step
 * @param step_description - further discription of the step
 * @param params - parameters that were send to the backend for this step
 */
export interface StepSummary {
    'step_id': string;
    step_idx: number;
    step_type: StepType;
    step_display_name: string;
    step_description: string;
    // TODO: in the future, we should extend the StepData interface for
    // each of the different steps, and type these more strongly!
    // Currently, we aren't sending this data!
    params?: Record<string, unknown>;
}

/**
 * The original location of a specific dataframe.
 */
export enum DFSource {
    Passed = 'passed',
    Imported = 'imported',
    Pivoted = 'pivoted',
    Merged = 'merged',
    Duplicated = 'duplicated',
    Transposed = 'transposed',
    Melted = 'melted',
}

/**
 * The two different ways to combine filters together
 */
export type Operator = 'And' | 'Or';

export type BooleanFilterCondition = 'boolean_is_true'
| 'boolean_is_false'

export type StringFilterCondition = 'contains'
| 'string_does_not_contain'
| 'string_exactly'
| 'string_not_exactly'
| 'string_starts_with'
| 'string_ends_with'

export type NumberFilterCondition = 'number_exactly'
| 'number_not_exactly'
| 'greater'
| 'greater_than_or_equal'
| 'less'
| 'less_than_or_equal'
| 'number_lowest'
| 'number_highest'

export type DatetimeFilterCondition = 'datetime_exactly'
| 'datetime_not_exactly'
| 'datetime_greater'
| 'datetime_greater_than_or_equal'
| 'datetime_less'
| 'datetime_less_than_or_equal'

export type SharedFilterCondition = 'empty'
| 'not_empty'
| 'most_frequent'
| 'least_frequent'


export interface FilterType {
    condition: BooleanFilterCondition | StringFilterCondition | NumberFilterCondition | DatetimeFilterCondition | SharedFilterCondition;
    value: string | number;
}
export interface StringValueFilterType {
    condition: BooleanFilterCondition | StringFilterCondition | NumberFilterCondition | DatetimeFilterCondition | SharedFilterCondition;
    value: string;
}


export interface FilterGroupType {
    filters: FilterType[];
    operator: Operator
}

export interface ColumnFilters {
    operator: 'Or' | 'And',
    filters: (FilterType | FilterGroupType)[]
}

export interface ColumnFilterMap {
    [column: string]: ColumnFilters;
}


export enum NumberColumnFormatEnum {
    PLAIN_TEXT = 'plain text',
    CURRENCY = 'currency',
    ACCOUNTING = 'accounting',
    PERCENTAGE = 'percentage',
    SCIENTIFIC_NOTATION = 'scientific notation'
}

export type ColumnFormatType = | 
{
    type?: NumberColumnFormatEnum,
    precision?: number
} // In the future, if we add other column format types, we add them here

export type ConditionalFormat = {
    // The format_uuid is a random string we use to uniquely identify different formats, so 
    // that we can easily figure out which formats are invalid between backend and frontend
    format_uuid: string, 
    columnIDs: ColumnID[],
    filters: FilterType[],
    color: string | undefined
    backgroundColor: string | undefined
}

export type DataframeFormat = {
    columns: Record<ColumnID, ColumnFormatType | undefined>,
    headers: {
        color?: string,
        backgroundColor?: string
    },
    rows: {
        even: {
            color?: string,
            backgroundColor?: string
        }
        odd: {
            color?: string,
            backgroundColor?: string
        }
    },
    border: {
        borderStyle?: string,
        borderColor?: string,
    },
    conditional_formats: ConditionalFormat[],
}


/**
 * @param noErrorModal set in the backend by some events, when we want
 * to pass the error directly through to the caller
 */
export interface MitoError {
    event: string;
    type: string;
    header: string;
    to_fix: string;
    traceback?: string;
}

/**
 * A column header can be a string a number, or a boolean, as well as a
 * list of these items.
 * 
 * If it is a list of these items, then this is a multi-index column header
 * and should be displayed as such.
 */
export type PrimitiveColumnHeader = string | number | boolean;
export type MultiIndexColumnHeader = PrimitiveColumnHeader[]; // TODO: is this a bug? Can we have a multi-index with a multi-index inside it
export type ColumnHeader = PrimitiveColumnHeader | MultiIndexColumnHeader;
export type DisplayColumnHeader = string // we alias this for type clarity and readability

export type ColumnID = string;
/**
 * A map from column IDs -> Column Headers
 */
export type ColumnIDsMap = Record<ColumnID, ColumnHeader>;

export type ConditionalFormattingResult = {
    'invalid_conditional_formats': Record<string, ColumnID[] | undefined>,
    'results': Record<ColumnID, Record<number | string, {color: string | undefined, backgroundColor: string | undefined} | undefined> | undefined>
}

/**
 * Data that will be displayed in the sheet itself.
 * 
 * @param dfName - the name of the dataframe
 * @param dfSource - the source of the dataframe
 * @param numRows - the number of rows in the data. Should be equal to data[0].length
 * @param numColumns - the number of columns in the data. Should be equal to data.length
 * @param data - a list of the columns to display in the sheet, including their id and header, their dtype, as well as a list of columnData (which is the actual data in this column)
 * @param columnIDsMap - for this dataframe, a map from column id -> column headers
 * @param columnSpreadsheetCodeMap - for this dataframe, a map from column id -> spreadsheet formula
 * @param columnFiltersMap - for this dataframe, a map from column id -> filter objects
 * @param columnDtypeMap - for this dataframe, a map from column id -> column dtype
 * @param index - the indexes in this dataframe
 */
export type SheetData = {
    dfName: string;
    dfSource: DFSource;
    numRows: number,
    numColumns: number,
    data: {
        columnID: ColumnID;
        columnHeader: ColumnHeader;
        columnDtype: string;
        columnData: (string | number | boolean)[];
    }[];
    columnIDsMap: ColumnIDsMap;
    columnSpreadsheetCodeMap: Record<ColumnID, string>;
    columnFiltersMap: ColumnFilterMap;
    columnDtypeMap: Record<ColumnID, string>;
    index: (string | number)[];
    dfFormat: DataframeFormat;
    conditionalFormattingResult: ConditionalFormattingResult;
};


export type GraphPreprocessingParams = {
    safety_filter_turned_on_by_user: boolean
}
export type GraphCreationParams<T> = {
    // Available to all graph types
    graph_type: GraphType,
    sheet_index: number,
    x_axis_column_ids: ColumnID[]
    y_axis_column_ids: ColumnID[]
    color: ColumnID | undefined
    facet_col_column_id: ColumnID | undefined
    facet_row_column_id: ColumnID | undefined,
    facet_col_wrap: T | undefined,
    facet_col_spacing: T | undefined,
    facet_row_spacing: T | undefined,
    
    // Paramaters that are only available for some graph types. 
    // To create these parameters, make sure to update the setGraphType and getDefaultGraphParams so that they 
    // set the param to undefined for all graph types that don't have the param.
    points: string | false | undefined
    line_shape: string | undefined,
    nbins: T | undefined
    histnorm: string | undefined,
    histfunc: string | undefined,
}
export type GraphStylingParams<T> = {
    title: {
        title: string | undefined, // when undefined, we use Ploty's default title
        visible: boolean,
        title_font_color: string, // defaults to #2f3e5d
    },
    xaxis: {
        title: string | undefined; // when undefined, we use Ploty's default title
        visible: boolean;
        title_font_color: string, // defaults to #2f3e5d
        type: string | undefined // when undefined, we use Plotly's default
        showgrid: boolean,
        gridwidth: T | undefined
        rangeslider: {
            visible: boolean,
        }
    },
    yaxis: {
        title: string | undefined, // when undefined, we use Ploty's default title
        visible: boolean
        title_font_color: string, // defaults to #2f3e5d
        type: string | undefined, // when undefined, we use Plotly's default
        showgrid: boolean,
        gridwidth: T | undefined,
    },
    showlegend: boolean,
    legend: {
        title: {
            text: string | undefined
        } | undefined
        orientation: 'v' | 'h' 
        x: T | undefined
        y: T | undefined
    }
    
    plot_bgcolor: string // The inner part of the plot with data background. Defaults to a blue-ish shade
    paper_bgcolor: string // The outer part of the plot around the data. Defaults to white

    // Paramaters that are only available for some graph types. 
    // To create these parameters, make sure to update the setGraphType and getDefaultGraphParams so that they 
    // set the param to undefined for all graph types that don't have the param.
    barmode: string | undefined,
    barnorm: string | undefined
}

type GraphParamsGeneric<T> = {
    graphPreprocessing: GraphPreprocessingParams,
    graphCreation: GraphCreationParams<T>,
    graphStyling: GraphStylingParams<T>,
};

/**
 * Data about all of the graphs. For each graph, it contains all of the parameters used to construct the graph,
 * the actual graph html & javascript, and the generated code.
 * 
 * @param graphParams - all of the parameters used to construct the graph
 * @param [graphOutput] - the python code, the graph html, and the graph script 
 */
type GraphDataGeneric<T> = {
    graphParams: GraphParamsGeneric<T>,
    graphOutput: GraphOutput, 
    graphTabName: string
};

// We have separate frontend and backend params so that we can 
// handle input fields which must be strings on the frontend to handle 
// decimal places and negative signs, while also allowing us to send 
// correctly types params to the backend.
export type GraphDataFrontend = GraphDataGeneric<string>
export type GraphDataBackend = GraphDataGeneric<number>
export type GraphParamsFrontend = GraphParamsGeneric<string>
export type GraphParamsBackend = GraphParamsGeneric<number>

export type GraphOutput = {
    graphGeneratedCode: string,
    graphHTML: string,
    graphScript: string,
} | undefined;

export type GraphID = string;

export type GraphDataDict = Record<GraphID, GraphDataFrontend>


export interface ConcatParams {
    join: 'inner' | 'outer',
    ignore_index: boolean,
    sheet_indexes: number[]
}

// NOTE: these aggregation functions need to be supported
// in mitosheet/steps/pivot.py as well
export enum AggregationType {
    COUNT = 'count', 
    COUNT_UNIQUE = 'count unique',
    SUM = 'sum',
    MEAN = 'mean',
    MEDIAN = 'median',
    STD = 'std',
    MIN = 'min',
    MAX = 'max', 
}

// The parameters saved on the backend
export interface BackendPivotParams {
    sheet_index: number,
    pivot_rows_column_ids: ColumnID[],
    pivot_columns_column_ids: ColumnID[],
    values_column_ids_map: Record<ColumnID, AggregationType[]>,
    flatten_column_headers: boolean;
    destination_sheet_index?: number;
}

// The parameters used by the frontend. The type of the params is different between the 
// backend and the frontend, due to it being easier to manipulate as an array on the 
// frontend while keeping the ordering for values
export interface FrontendPivotParams {
    sourceSheetIndex: number,
    pivotRowColumnIDs: ColumnID[],
    pivotColumnsColumnIDs: ColumnID[],
    // NOTE: storing these values as an array makes keeping an order of them
    // much much easier, and generally is the way to do it!
    pivotValuesColumnIDsArray: [ColumnID, AggregationType][],
    flattenColumnHeaders: boolean,
    destinationSheetIndex?: number
}


/**
 * The current viewport of the sheet. As not all data is rendered, this object
 * marks what is actually rendered. Note that not all of it may be visible to 
 * the user.
 * 
 * @param startingRowIndex - The index of the first row to be rendered.
 * @param numRowsRendered - the number of rows in the data. Should be equal to data.length
 * @param startingColumnIndex - The index of the first column to be rendered.
 * @param numColumnsRendered - The number of columns to actually render. Some of them may not be visible.
 */
export interface SheetView {
    startingRowIndex: number;
    numRowsRendered: number;
    startingColumnIndex: number;
    numColumnsRendered: number;
}

/**
 * The current selection that the user has made in the sheet. Each of these values
 * can range from -1 to the maximum index of a column or a row (one less than the 
 * length of the sheet data). 
 * 
 * @remarks If the columnIndex is -1, then this is the IndexHeaders. If the rowIndex is -1, 
 * then this is the ColumnHeaders. This makes it easy to natually move selection
 * around between grid data and the headers.
 * 
 * @remarks The ending indexes can be less than the starting index, as the selection 
 * also encodes information about the direction that the user has done the selection.
 * 
 * @param startingRowIndex - The index of the row that the user first selected.
 * @param endingRowIndex - The index of the final row the user is selecting. Might be less than startingRowIndex.
 * @param startingColumnIndex - The index of the column that the user first selected.
 * @param endingColumnIndex - The index of the final column the user is selecting. Might be less than startingColumnIndex.
 */
export interface MitoSelection {
    startingRowIndex: number;
    endingRowIndex: number;
    startingColumnIndex: number;
    endingColumnIndex: number;
}

/**
 * The border style to be applied to a range of cells.
 * 
 * @param borderRight - The right border style
 * @param borderLeft - The left border style
 * @param borderTop - The top border style
 * @param borderBottom - The bottom border style
 */
export interface BorderStyle {
    borderRight?: string | undefined
    borderLeft?: string | undefined
    borderTop?: string | undefined
    borderBottom?: string | undefined
}

/**
 * The amount grid data, column headers, and index headers should
 * translate so that they are properly placed.
 * 
 * @param x - How much the renderer should translate in the x
 * @param y - How much the renderer should translate in the y
 */
export interface RendererTranslate {
    x: number;
    y: number;
}

/**
 * Stores the height and width of some div.
 * 
 * @param width - Width of the object represented
 * @param height - Height of the object represented
 */
export interface Dimension {
    width: number;
    height: number;
}

/**
 * Stores the amount the content inside a div has been
 * scrolled.
 * 
 * @param scrollTop - a measurement of the distance from the element's top to its topmost visible content.
 * @param scrollLeft - a measurement of the distance from the element's left to its topmost visible content.
 */
export interface ScrollPosition {
    scrollTop: number;
    scrollLeft: number;
}

/**
 * Stores the current state of the editor, keeping track of all the information
 * to allow the user to edit the cell. As we don't use a browser input, we have to 
 * keep track of more than usual.
 * 
 * The editor might be displayed as a cell editor or a column header editor, depending
 * on what exactly is being edited!
 * 
 * @param rowIndex - Row index of the cell being edited
 * @param columnIndex - Column index of the cell being edited
 * @param formula - The current formula. This might not be what is displayed to the user, if they have pendingSelectedColumns
 * @param pendingSelectedColumns - A list of columns that the user has selected through the arrow keys or clicking on columns. Also stores _where_ in the formula these columns should be inserted
 * @param arrowKeysScrollInFormula - The user can click on the editor to make the arrow keys scroll in the editor rather than in the sheet
 * @param editorLocation -- The location of the cell editor, either a cell or formula bar
 */
export type EditorState = {
    rowIndex: number;
    columnIndex: number;
    formula: string;
    editingMode: 'set_column_formula' | 'set_cell_value';

    pendingSelectedColumns?: {
        columnHeaders: (ColumnHeader)[]
        selectionStart: number,
        selectionEnd: number,
    };

    /* 
        Represents where the arrow keys should scroll, if the user
        types something. 
    */
    arrowKeysScrollInFormula?: boolean;
    editorLocation: 'cell' | 'formula bar';
};

/**
 * Stores the width data of the columns in the sheet.
 * 
 * @param widthArray - At each index, stores the width of the column at that index, in pixels
 * @param widthSumArray - For efficiency, stores the _sums_ of the columns up to and including that index.
 * @param totalWidth - The total sum of all the widths of all the columns
 */
export interface WidthData {
    widthArray: number[];
    widthSumArray: number[];
    totalWidth: number;
}

/**
 * Stores the width data of the columns in the sheet.
 * The state of the grid is represented by a viewport with a height and width,
 * the scroll position of the scroller div in the viewport, and the selected 
 * range of cells within the grid.
 * 
 * We default the selection to -2s so that nothing is selected and all our selection
 * code doesn't need to handle too many special cases.
 * 
 * We store all of this state in a single object as if any of them change, the entire
 * grid (the data, the headers) needs to rerender. Thus, we don't want to set them
 * indivigually so that we can limit the amount of unnecessary rerendering we do.
 * 
 * Only put state in here that causes a rerender of the entire grid when any element changes
 * and is consistently passed as props to the grid and headers.
 * 
 * @param sheetIndex - The sheet that this grid state represents
 * @param viewport - The size of the viewport
 * @param scrollPosition - Scroll position in the grid
 * @param selections - Selected ranges
 * @param copiedSelections - The ranges that currently have been copied
 * @param columnIDsArray - A mapping from sheetIndex -> columnIndex -> columnID
 * @param widthDataArray - A list of width data for each sheet
 */
export interface GridState {
    sheetIndex: number;
    viewport: Dimension;
    scrollPosition: ScrollPosition;
    selections: MitoSelection[];
    copiedSelections: MitoSelection[];
    columnIDsArray: ColumnID[][];
    widthDataArray: WidthData[];
}

/**
 * The type of data that is in this current Mito analysis.
 * 
 * @remark this should be the same as the file in the Python code
 * which is in data_in_mito.py
 */
export enum DataTypeInMito {
    NONE = 'none',
    PROVIDED = 'provided',
    TUTORIAL = 'tutorial',
    PERSONAL = 'personal',
}



export type ExperimentID = 
    | 'title_name';

interface Experiment {
    experiment_id: ExperimentID;
    variant: "A" | "B";
}


/**
 * An object representing all the data about the analysis that is occuring currently.
 * 
 * @param analysisName - the name of the analysis id that is for writing to the cell (after the analysis has been replayed)
 * @param analysisToReplay - the analysis that was passed through the analysis_to_replay parameter to the mitosheet.sheet call
 * @param code - the transpiled code of this analysis
 * @param stepSummaryList - a list of step summaries for the steps in this analysis
 * @param currStepIdx - the index of the currently checked out step, in the stepSummaryList
 * @param dataTypeInTool - the type of data in the tool in this analysis
 * @param graphDataDict - a mapping from graphID to all of the relevant graph information
 * @param updateEventCount - the number of update events that have been successfully processed by the frontend
 * @param renderCount - the number of times this sheet has rendered. Note that this is per instance of the backend object, 
 *        so it increments by one when the page the sheet is rendered on is refreshed, but resets to zero if a new
 *        mitosheet.sheet() call is made (even if it replays the analysis), as this is a new backend object.
 * @param lastResult - This is the result of the last step that was applied. This might be undefined if the 
 *        step does not return a result
 * @param experiment - The experiment that this user is currently running, which may not be defined
 * @param savedAnalysisJSON - The string that is saved in the analysis file that is stored on disk, and in the notebook metadata
 */
export interface AnalysisData {
    analysisName: string,
    analysisToReplay: {
        analysisName: string,
        existsOnDisk: boolean,
    } | null | undefined,
    code: string[],
    stepSummaryList: StepSummary[],
    currStepIdx: number,
    dataTypeInTool: DataTypeInMito;
    graphDataDict: GraphDataDict;
    updateEventCount: number;
    renderCount: number;
    lastResult: any;
    experiment: Experiment | undefined;
    savedAnalysisJSON: string,
}

export interface SavedAnalysis {
    'version': string,
    'author_hash': string,
    'steps_data': {step_version: number, step_type: StepType, params: Record<string, string>}[],
}
/**
 * An object represending this user
 * 
 * @param userEmail - the email of the user. May be an empty string if they have not signed up yet
 * @param receivedTours - a list of the tours they have received
 * @param isPro - if the user is a pro user
 * @param excelImportEnabled - if the user has the necessary packages optional dependencies, and Python and Pandas version to import Excel files.
 * @param telemetryEnabled - if the user has telemetry enabled
 * @param isLocalDeployment - if the user is deployed locally or not
 * @param shouldUpgradeMitosheet - if the user should upgrade their mitosheet
 * @param numUsages - the number of times the user has used the tool (maxes out at 50 currently)
 * @param usageTriggeredFeedbackID - the id of the usage triggered feedback id to display to the user
 */
export interface UserProfile {
    userEmail: string;
    receivedTours: string[];
    receivedChecklists: Record<ChecklistID, string[] | undefined>;

    isPro: boolean;
    excelImportEnabled: boolean;
    telemetryEnabled: boolean;
    isLocalDeployment: boolean;
    shouldUpgradeMitosheet: boolean;
    numUsages: number;
}


/**
 * The different functions necessary to update the state of the Mito component
 * from outside the component (e.g. after receiving a message).
 */
export interface MitoStateUpdaters {
    setSheetDataArray: React.Dispatch<React.SetStateAction<SheetData[]>>,
    setAnalysisData: React.Dispatch<React.SetStateAction<AnalysisData>>,
    setUserProfile: React.Dispatch<React.SetStateAction<UserProfile>>,
    setUIState: React.Dispatch<React.SetStateAction<UIState>>,
}

export interface CSVExportState { exportType: 'csv' }
export interface ExcelExportState { exportType: 'excel', sheetIndexes: number[] }

export type ToolbarDropdowns = 'Edit' | 'Dataframes' | 'Columns' | 'Rows' | 'Graphs' | 'Format' | 'View' | 'Help';

/**
 * State of the UI, all in one place for ease.
 */
export interface UIState {
    loading: [string, string | undefined, string][]; // message id, step id (if it exists), message type
    currOpenModal: ModalInfo;
    currOpenTaskpane: TaskpaneInfo;
    selectedColumnControlPanelTab: ControlPanelTab;
    exportConfiguration: CSVExportState | ExcelExportState;
    selectedSheetIndex: number;
    selectedGraphID: GraphID | undefined;
    selectedTabType: 'data' | 'graph';
    currOpenToolbarDropdown: undefined | ToolbarDropdowns;
    toolbarDropdown: 'import' | 'format' | 'dtype' | undefined;
}

/**
 * Used to identify the feedback that the user is prompted for. 
 * When we add new feedback options, add it here!
 */
export const enum FeedbackID {
    COMPANY = 'company/organization',
    SOURCE = 'source',
    MITO_GOAL = 'mito_goal',
}

/*
    ActionEnum is used to identify a specific action.

    Listed in alphabetical order: first by non-spreadsheet functions,
    then by spreadsheet functions
*/
export enum ActionEnum {
    Add_Column = 'add column',
    Catch_Up = 'catch up',
    Clear = 'clear',
    Change_Dtype = 'change dtype',
    Column_Summary = 'column summary',
    Copy = 'copy',
    Delete_Column = 'delete column',
    Delete_Dataframe = 'delete dataframe',
    Delete_Graph = 'delete graph',
    Delete_Row = 'delete row',
    Drop_Duplicates = 'drop duplicates',
    Duplicate_Dataframe = 'duplicate dataframe',
    Duplicate_Graph = 'duplicate graph',
    Docs = 'docs',
    Export = 'export',
    Fill_Na = 'fill na',
    Filter = 'filter',
    Format_Number_Columns = 'format number columns',
    Fullscreen = 'fullscreen',
    Graph = 'graph',
    Help = 'help',
    Import_Dropdown = 'import dropdown',
    Import_Files = 'import files',
    Merge = 'merge',
    Concat_Dataframes = 'concat_dataframes', // Note the unfortunate overlap with concat
    Pivot = 'pivot',
    Precision_Increase = 'precision increase',
    Precision_Decrease = 'precision decrease',
    Promote_Row_To_Header = 'promote row to header',
    Redo = 'redo',
    Rename_Column = 'rename column',
    Rename_Dataframe = 'rename dataframe',
    Rename_Graph = 'rename graph',
    See_All_Functionality = 'see all functionality',
    //Search = 'search',
    Set_Cell_Value = 'set cell value',
    Set_Column_Formula = 'set column formula',
    Sort = 'sort',
    Split_Text_To_Column = 'split text to column',
    Steps = 'steps',
    Undo = 'undo',
    Unique_Values = 'unique values',
    Upgrade_To_Pro = 'upgrade to pro',
    Transpose = 'transpose',
    Melt = 'melt',
    One_Hot_Encoding = 'one_hot_encoding',
    Set_Dataframe_Format = 'set_dataframe_format',
    Conditional_Formatting = 'ConditionalFormatting',
    Dataframe_Import = 'Dataframe_Import',
    UPDATEIMPORTS = 'updateImports',
    // AUTOGENERATED LINE: ACTIONENUM (DO NOT DELETE)
    /* Spreadsheet Formulas Section */
    ABS = 'abs',
    AND = 'and',
    AVG = 'avg',
    BOOL = 'bool',
    CLEAN = 'clean',
    CONCAT = 'concat',
    CORR = 'corr',
    DATEVALUE = 'datevalue',
    DAY = 'day',
    ENDOFBUSINESSMONTH = 'endofbusinessmonth',
    ENDOFMONTH = 'endofmonth',
    EXP = 'exp',
    FILLNAN = 'fillnan',
    FIND = 'find',
    FLOAT = 'float',
    HOUR = 'hour',
    IF = 'if',
    INT = 'int',
    KURT = 'kurt',
    LEFT = 'left',
    LEN = 'len',
    LOWER = 'lower',
    MAX = 'max',
    MID = 'mid',
    MIN = 'min',
    MINUTE = 'minute',
    MONTH = 'month',
    MULTIPLY = 'multiply',
    OR = 'or',
    POWER = 'power',
    PROPER = 'proper',
    QUARTER = 'quarter',
    RIGHT = 'right',
    ROUND = 'round',
    SECOND = 'second',
    SKEW = 'skew',
    STARTOFBUSINESSMONTH = 'startofbusinessmonth',
    STARTOFMONTH = 'startofmonth',
    STRIPTIMETOMINUTES = 'striptimetominutes',
    STRIPTIMETOHOURS = 'striptimetohours',
    STRIPTIMETODAYS = 'striptimetodays',
    STRIPTIMETOMONTHS = 'striptimetomonths',
    STRIPTIMETOYEARS = 'striptimetoyears',
    SUBSTITUTE = 'substitute',
    SUM = 'sum',
    TEXT = 'text',
    TRIM = 'trim',
    TYPE = 'type',
    UPPER = 'upper',
    VALUE = 'value',
    VAR = 'var',
    WEEK = 'week',
    WEEEKDAY = 'weekday',
    YEAR = 'year',
}

// These actions should be used in the toolbar, in search, and 
// anywhere else the action is performed. That is why we call it Action
export interface Action {
    // We store the type of the action so that we can introspect the type of the variable
    type: ActionEnum;

    // The short title for the action. Should be title case, as you want to display it.
    shortTitle: string

    // The optional long title for the action.
    longTitle: string

    /* 
        The function to call if the action is taken by the user. This should
        not require any parameters to be called, as all the necessary data
        for the calling of this action should be created when it is constructed
        NOTE: this no param usage is what we do in the toolbar, and it works
        really well to allow the caller to be able to use this action without
        having to know anything about it
    */
    actionFunction: () => void;

    // A list of the search terms that can be used to discover this action
    searchTerms: string[]

    /*
        The function to call to determine if the action is enabled or not,
        which returns a string with the reason for being disabled, or undefined
        if the action is not disabled. 

        Some basic rules of thumb:
        1.  An action that opens an interface that lets the user select a datasource is only
            disabled if no sheet exists in Mito
        2.  An action that relies on a specific column, is only disabled if that column does not exist. 
        3.  An action that does not rely on the state of Mito (ie: opening full screen mode) is never disabled.
    */
    isDisabled: () => string | undefined

    // The tooltip to display in the toolbar or search bar when this is hovered over
    tooltip: string

    // Optionally categorize the action, so it can easily be sorted later
    category?: 'spreadsheet formula'

    // If this action has a keyboard shortcut, then you can display this by setting these values
    displayKeyboardShortcuts?: {
        mac: string,
        windows: string
    }
}

export enum GraphSidebarTab {
    Setup = 'setup',
    Style = 'style',
    Export = 'export'
}


// A fancy type taken from here: https://stackoverflow.com/questions/41980195/recursive-partialt-in-typescript
export type RecursivePartial<T> = {
    [P in keyof T]?:
    T[P] extends (infer U)[] ? RecursivePartial<U>[] :
        T[P] extends Record<string, unknown> ? RecursivePartial<T[P]> :
            T[P];
};