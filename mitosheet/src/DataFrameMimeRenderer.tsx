import { IRenderMime } from '@jupyterlab/rendermime-interfaces';
import { Widget } from '@lumino/widgets';
import { ReactWidget } from '@jupyterlab/apputils';
import * as React from 'react';
import { INotebookTracker } from '@jupyterlab/notebook';
import { getLastNonEmptyLine } from './jupyter/code';
import { getCellAtIndex, getCellText } from './jupyter/extensionUtils';

const CLASS_NAME = 'jp-DataFrameViewer';

// React component to display the output from Python
const SpreadsheetDataframeComponent = (props: { htmlContent: string, jsCode?: string }) => {
    // Hook to run the JavaScript code after the component is mounted
    React.useEffect(() => {
        if (props.jsCode) {
            // Dynamically create a script tag to execute the JS code
            const script = document.createElement('script');
            script.type = 'text/javascript';
            script.text = props.jsCode;
            document.body.appendChild(script);
            console.log("JavaScript code injected and executed.");

            // Clean up the script after execution
            return () => {
                document.body.removeChild(script);
            };
        }
    }, [props.jsCode]);

    return (
        <div
            dangerouslySetInnerHTML={{ __html: props.htmlContent }}
        ></div>
    );
};

export class DataFrameMimeRenderer extends Widget implements IRenderMime.IRenderer {
    private _notebookTracker: INotebookTracker;
    private _defaultRenderer: IRenderMime.IRenderer;

    constructor(options: IRenderMime.IRendererOptions, notebookTracker: INotebookTracker, defaultRenderer: IRenderMime.IRenderer) {
        super();
        this.addClass(CLASS_NAME);
        this._notebookTracker = notebookTracker;
        this._defaultRenderer = defaultRenderer;
    }

    async renderModel(model: IRenderMime.IMimeModel): Promise<void> {
        const originalRawData = model.data['text/html']?.toString();
        const isDataframeOutput = originalRawData?.includes('class="dataframe"');
        const notebook = this._notebookTracker.currentWidget?.content;
        const cells = notebook?.model?.cells;
        const activeCellIndex = notebook?.activeCellIndex

        let dataframeVariableName = undefined;
        let cellID = undefined
        if (activeCellIndex) {
            const previousCell = getCellAtIndex(cells, activeCellIndex - 1)
            dataframeVariableName = getLastNonEmptyLine(getCellText(previousCell))
            cellID = previousCell?.id
        }
        if (isDataframeOutput) {
            console.log('Dataframe detected!!!!');

            // Define the Python code to run
            const pythonCode = `mitosheet.sheet(${dataframeVariableName || ''}, cell_id='${cellID}')`;

            try {
                const notebookPanel = this._notebookTracker.currentWidget;
                const kernel = notebookPanel?.context.sessionContext.session?.kernel;

                if (!kernel) {
                    console.error('No active kernel found.');
                    return;
                }

                // Execute the Python code
                const future = kernel.requestExecute({ code: pythonCode });

                future.onIOPub = (msg: any) => {
                    if (msg.header.msg_type === 'execute_result' || msg.header.msg_type === 'display_data') {
                        const htmlOutput = msg.content.data['text/html'];
                        if (htmlOutput) {
                            // Extract the javascript code so we can execute it when we render the output.
                            // Remember, that the javascript code is actually what creates the sheet interface!
                            const scriptMatch = htmlOutput.match(/<script[^>]*>([\s\S]*?)<\/script>/);
                            const jsCode = scriptMatch ? scriptMatch[1] : '';

                            // Create a React widget to display the HTML output
                            const reactWidget = ReactWidget.create(
                                <SpreadsheetDataframeComponent htmlContent={htmlOutput} jsCode={jsCode} />
                            );

                            // Attatch the Mito widget to the node
                            this.node.innerHTML = '';
                            Widget.attach(reactWidget, this.node);
                        }
                    }
                };

                future.done.then(() => {
                    console.log('Python code executed successfully.');
                })

            } catch (error) {
                console.error('Error executing Python code:', error);
                this.node.innerHTML = `<div style="color: red;">Error rendering sheet</div>`;
            }

        } else {
            await this._defaultRenderer.renderModel(model);
            this.node.appendChild(this._defaultRenderer.node);
        }

        return Promise.resolve();
    }

    dispose(): void {
        super.dispose();
    }
}

export default DataFrameMimeRenderer;