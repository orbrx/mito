import React from 'react';
import PythonCode from './PythonCode';
import { INotebookTracker } from '@jupyterlab/notebook';
import { getNotebookName, writeCodeToActiveCell } from '../../../utils/notebook';
import { IRenderMimeRegistry } from '@jupyterlab/rendermime';
import { removeMarkdownCodeFormatting } from '../../../utils/strings';
import { JupyterFrontEnd } from '@jupyterlab/application';
import { OperatingSystem } from '../../../utils/user';
import '../../../../style/CodeMessagePart.css'


interface ICodeBlockProps {
    code: string
    role: 'user' | 'assistant'
    rendermime: IRenderMimeRegistry
    notebookTracker: INotebookTracker,
    app: JupyterFrontEnd,
    isLastAiMessage: boolean,
    operatingSystem: OperatingSystem,
    setDisplayCodeDiff: React.Dispatch<React.SetStateAction<boolean>>;
}

const CodeBlock: React.FC<ICodeBlockProps> = ({
    code, 
    role, 
    rendermime, 
    notebookTracker, 
    app, 
    isLastAiMessage,
    operatingSystem,
    setDisplayCodeDiff
}): JSX.Element => {
    
    const notebookName = getNotebookName(notebookTracker)

    const copyCodeToClipboard = () => {
        const codeWithoutMarkdown = removeMarkdownCodeFormatting(code)
        navigator.clipboard.writeText(codeWithoutMarkdown)
    }

    if (role === 'user') {
        return (
            <div className='code-message-part-container'>
                <PythonCode
                    code={code}
                    rendermime={rendermime}
                />
            </div>
        )
    }

    

    if (role === 'assistant') {
        return (
            <div className='code-message-part-container'>
                <div className='code-message-part-toolbar'>
                    <div className='code-location'>
                        {notebookName}
                    </div>
                    <button onClick={() => writeCodeToActiveCell(notebookTracker, code, true)}>Apply to cell {isLastAiMessage ? (operatingSystem === 'mac' ? 'CMD+Y' : 'CTRL+Y') : ''}</button>
                    <button onClick={() => setDisplayCodeDiff(displayCodeDiff => !displayCodeDiff)}>
                        Display Diff
                    </button>
                    <button onClick={copyCodeToClipboard}>Copy</button>
                </div>
                <PythonCode
                    code={code}
                    rendermime={rendermime}
                />
            </div>
        )
    }

    return <></>
}

export default CodeBlock