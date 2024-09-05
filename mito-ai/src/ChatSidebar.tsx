import React from 'react';
import { ReactWidget } from '@jupyterlab/apputils';
import Chat from './Chat';
import { INotebookTracker } from '@jupyterlab/notebook';
import { IRenderMimeRegistry } from '@jupyterlab/rendermime';
import { LabIcon } from '@jupyterlab/ui-components';
import chatIconSvg from '../src/icons/ChatIcon.svg';

export const chatIcon = new LabIcon({
    name: 'mito_ai',
    svgstr: chatIconSvg
});

export function buildChatSidebar(
    notebookTracker: INotebookTracker,
    rendermime: IRenderMimeRegistry
) {
    const chatWidget = ReactWidget.create(
        <Chat 
            notebookTracker={notebookTracker}
            rendermime={rendermime}
        />
    ) 
    chatWidget.id = 'mito_ai';
    chatWidget.title.icon = chatIcon;
    chatWidget.title.caption = 'AI Chat for your JupyterLab';
    chatWidget.addClass('chat-widget');

    return chatWidget;
}