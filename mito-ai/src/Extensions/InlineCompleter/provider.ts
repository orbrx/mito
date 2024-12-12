import { Notification, showErrorMessage } from '@jupyterlab/apputils';
import { IEditorMimeTypeService } from '@jupyterlab/codeeditor';
import {
  IEditorLanguageRegistry,
  type IEditorLanguage
} from '@jupyterlab/codemirror';
import {
  InlineCompletionTriggerKind,
  type CompletionHandler,
  type IInlineCompletionContext,
  type IInlineCompletionItem,
  type IInlineCompletionList,
  type IInlineCompletionProvider
} from '@jupyterlab/completer';
import { DocumentWidget } from '@jupyterlab/docregistry';
import type { ISettingRegistry } from '@jupyterlab/settingregistry';
import { PromiseDelegate, type JSONValue } from '@lumino/coreutils';
import type { IDisposable } from '@lumino/disposable';
import { Signal, Stream } from '@lumino/signaling';
import type OpenAI from 'openai';
import {
  CompletionWebsocketClient,
  type ICompletionWebsocketClientOptions
} from './client';
import type {
  CompletionError,
  ICompletionStreamChunk,
  InlineCompletionStreamChunk
} from './models';

/**
 * Mito AI inline completer
 *
 * It uses a WebSocket connection to request an AI model.
 */
export class MitoAIInlineCompleter
  implements IInlineCompletionProvider, IDisposable
{
  private _client: CompletionWebsocketClient;
  private _counter = 0;
  private _isDisposed = false;
  private _languageRegistry: IEditorLanguageRegistry;
  private _settings: MitoAIInlineCompleter.ISettings =
    MitoAIInlineCompleter.DEFAULT_SETTINGS;
  // Store only one inline completion stream
  //   Each new request should invalidate any other suggestions.
  private _currentPrefix = '';
  private _currentToken = '';
  private _currentStream: Stream<
    MitoAIInlineCompleter,
    InlineCompletionStreamChunk
  > | null = null;
  /**
   * Block processing chunks while waiting for the acknowledge request
   * that will provide the unique completion token.
   */
  private _completionLock = new PromiseDelegate<void>();
  private _fullCompletionMap = new WeakMap<
    Stream<MitoAIInlineCompleter, InlineCompletionStreamChunk>,
    string
  >();

  constructor({
    languageRegistry,
    ...clientOptions
  }: MitoAIInlineCompleter.IOptions) {
    this._languageRegistry = languageRegistry;
    this._client = new CompletionWebsocketClient(clientOptions);

    this._client
      .initialize()
      .then(() => {
        this._client.stream.connect(this._receiveStreamChunk, this);
        this._completionLock.resolve();
      })
      .catch(reason => {
        this._completionLock.reject(reason);
        console.error(
          'Failed to initialize the websocket connection for ai completions.',
          reason
        );
      });
  }

  /**
   * Completer unique identifier
   */
  readonly identifier: string = 'mito-ai';

  /**
   * Completer name
   */
  readonly name: string = 'Mito AI';

  /**
   * Whether the completer is disposed or not.
   */
  get isDisposed(): boolean {
    return this._isDisposed;
  }

  /**
   * Settings schema contributed by provider for user customization.
   */
  get schema(): ISettingRegistry.IProperty {
    return {
      properties: {
        triggerKind: {
          title: 'Inline completions trigger',
          type: 'string',
          oneOf: [
            { const: 'any', title: 'Automatic (on typing or invocation)' },
            { const: 'manual', title: 'Only when invoked manually' }
          ],
          description: 'When to trigger inline completions when using mito-ai.'
        }
      },
      default: MitoAIInlineCompleter.DEFAULT_SETTINGS as any
    };
  }

  /**
   * Callback on user settings changes.
   */
  async configure(settings: { [property: string]: JSONValue }): Promise<void> {
    this._settings = settings as unknown as MitoAIInlineCompleter.ISettings;
  }

  /**
   * Dispose of the resources used by the completer.
   */
  dispose(): void {
    if (this._isDisposed) {
      return;
    }
    this._isDisposed = true;
    this._client.stream.disconnect(this._receiveStreamChunk, this);
    this._client.dispose();
    this._resetCurrentStream();
    Signal.clearData(this);
  }

  /**
   * The method called when user requests inline completions.
   *
   * The implicit request (on typing) vs explicit invocation are distinguished
   * by the value of `triggerKind` in the provided `context`.
   */
  async fetch(
    request: CompletionHandler.IRequest,
    context: IInlineCompletionContext
  ): Promise<IInlineCompletionList<IInlineCompletionItem>> {
    if (!this.isEnabled()) {
      return Promise.reject('Mito AI completion is disabled.');
    }

    if (this.isDisposed) {
      return Promise.reject('Mito AI provider is disposed.');
    }

    // Block processing chunks while waiting for the acknowledge request
    // that will provide the unique completion token.
    await this._completionLock.promise;

    this._completionLock = new PromiseDelegate<void>();
    try {
      // Stop current stream if any
      this._resetCurrentStream();

      const allowedTriggerKind = this._settings.triggerKind;
      const triggerKind = context.triggerKind;
      if (
        allowedTriggerKind === 'manual' &&
        triggerKind !== InlineCompletionTriggerKind.Invoke
      ) {
        // Short-circuit if user requested to only invoke inline completions
        // on manual trigger. Users may still get completions
        // from other (e.g. less expensive or faster) providers.
        return {
          items: []
        };
      }
      const messageId = ++this._counter;

      const prefix = this._getPrefix(request);
      const result = await this._client.sendMessage({
        message_id: messageId.toString(),
        stream: true,
        type: 'inline_completion',
        messages: this._getInlineCompletionMessages(request, context, prefix)
      });

      this._currentPrefix = getPrefixLastLine(prefix);
      if (result.items[0]?.token) {
        this._currentToken = result.items[0].token;
        this._currentStream = new Stream<
          MitoAIInlineCompleter,
          InlineCompletionStreamChunk
        >(this);
      }

      const error = result.error;
      if (error) {
        this._notifyCompletionFailure(error);
        throw new Error(
          `Inline completion failed: ${error.type}\n${error.traceback}`
        );
      }

      return { items: result.items };
    } finally {
      this._completionLock.resolve();
    }

    function getPrefixLastLine(prefix: string) {
      for (let index = prefix.length - 1; index >= 0; index--) {
        if (prefix[index] === '\n') {
          return prefix.slice(index + 1);
        }
      }
      return prefix;
    }
  }

  /**
   * Whether the completer is enabled or not.
   */
  isEnabled(): boolean {
    return this._settings.enabled;
  }

  /**
   * Stream a reply for completion identified by given `token`.
   */
  async *stream(token: string): AsyncGenerator<{
    response: IInlineCompletionItem;
  }> {
    if (!this.isEnabled()) {
      throw new Error('Mito AI completion is disabled.');
    }
    // Wait for the acknowledge request to be fulfilled before starting the stream
    await this._completionLock.promise;

    if (this._currentToken !== token) {
      // New completion may be triggered before the code have started streaming
      // the previous one. So not raising an error here.
      console.debug(`No stream found for token '${token}'.`);
      return;
    }

    for await (const chunk of this._currentStream!) {
      // If a new completion is triggered, stop the current stream
      // before the backend has finished streaming the full suggestion.
      if (this._currentToken !== token) {
        break;
      }
      yield chunk;
      if (chunk.done || chunk.error) {
        // Break this for loop
        this._currentStream?.stop();
      }
    }
  }

  private _getInlineCompletionMessages(
    request: CompletionHandler.IRequest,
    context: IInlineCompletionContext,
    prefix: string
  ): OpenAI.Chat.ChatCompletionMessageParam[] {
    const mime = request.mimeType ?? IEditorMimeTypeService.defaultMimeType;
    const editorLanguage = this._languageRegistry.findByMIME(mime);

    let path = context.session?.path;
    if (!path && context.widget instanceof DocumentWidget) {
      path = context.widget.context.path;
    }
    const suffix = this._getSuffix(request);

    const language = editorLanguage
      ? this._resolveLanguage(editorLanguage)
      : '';
    const languageNote = language ? ` and written in ${language}` : '';

    const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
      {
        role: 'system',
        content: `
You are an application built to provide helpful code completion suggestions.
You should only produce code. Keep comments to minimum, use the
programming language comment syntax. Produce clean executable code.
The code is written for a data analysis and code development
environment which can execute code to produce graphics, tables and
interactive outputs.`
      },
      {
        role: 'user',
        content: `The document is called \`${path}\`${languageNote}.`
      },
      {
        role: 'user',
        content:
          'Complete the following code responding only with additional code, code comments or docstrings, and with no markdown formatting.'
      },
      {
        role: 'user',
        content: prefix
      }
    ];

    if (suffix) {
      messages.push(
        {
          role: 'user',
          content: 'The new code appears before the following snippet.'
        },
        {
          role: 'user',
          content: suffix
        }
      );
    }

    return messages;
  }

  /**
   * Extract prefix from request, accounting for context window limit.
   *
   * For the case of a cell, this extract all the code of the current cell
   * before the cursor.
   */
  private _getPrefix(request: CompletionHandler.IRequest): string {
    return request.text.slice(0, request.offset);
  }

  /**
   * Extract suffix from request, accounting for context window limit.
   *
   * For the case of a cell, this extract all the code of the current cell
   * after the cursor.
   */
  private _getSuffix(request: CompletionHandler.IRequest): string {
    return request.text.slice(request.offset);
  }

  private _notifyCompletionFailure(error: CompletionError) {
    Notification.emit(`Inline completion failed: ${error.type}`, 'error', {
      autoClose: false,
      actions: [
        {
          label: 'Show Traceback',
          callback: () => {
            showErrorMessage('Inline completion failed on the server side', {
              message: error.traceback
            });
          }
        }
      ]
    });
  }

  /**
   * Process the stream chunk to make it available in the awaiting generator.
   */
  private _receiveStreamChunk(
    _emitter: CompletionWebsocketClient,
    chunk: ICompletionStreamChunk
  ) {
    if (chunk.error) {
      this._notifyCompletionFailure(chunk.error);
    }

    const token = chunk.chunk.token;
    if (!token) {
      throw Error('Stream chunks must define `token` in `chunk`.');
    }

    if (this._currentToken !== token) {
      // This may happen if the backend is still streaming for a previous token
      console.debug(
        `Received completion chunk for an unknown token '${token}'`
      );
      return;
    }

    if (!this._currentStream) {
      throw Error(`Stream not found for token ${token}`);
    }

    let fullCompletion = this._fullCompletionMap.get(this._currentStream) ?? '';
    fullCompletion += chunk.chunk.insertText;
    this._fullCompletionMap.set(this._currentStream, fullCompletion);

    // Clean suggestion
    let cleanedCompletion = fullCompletion.slice(0);
    if (this._currentPrefix) {
      if (
        cleanedCompletion.startsWith(this._currentPrefix) ||
        this._currentPrefix.startsWith(cleanedCompletion)
      ) {
        cleanedCompletion = cleanedCompletion.slice(this._currentPrefix.length);
      } else if (!['\n', ' '].includes(cleanedCompletion[0])) {
        cleanedCompletion = '\n' + cleanedCompletion;
      }
    }

    this._currentStream.emit({
      done: chunk.done,
      error: chunk.error,
      parent_id: chunk.parent_id,
      response: {
        insertText: cleanedCompletion,
        isIncomplete: !chunk.done,
        error: chunk.chunk.error,
        token: chunk.chunk.token
      },
      type: chunk.type
    });
  }

  private _resetCurrentStream() {
    this._currentToken = '';
    if (this._currentStream) {
      this._currentStream.stop();
      this._fullCompletionMap.delete(this._currentStream);
      this._currentStream = null;
    }
  }

  private _resolveLanguage(language: IEditorLanguage | null) {
    if (!language) {
      return 'plain English';
    }
    if (language.name === 'ipython') {
      return 'python';
    } else if (language.name === 'ipythongfm') {
      return 'markdown';
    }
    return language.name;
  }
}

export namespace MitoAIInlineCompleter {
  export interface IOptions extends ICompletionWebsocketClientOptions {
    /**
     * CodeMirror language registry.
     */
    languageRegistry: IEditorLanguageRegistry;
  }

  export interface ISettings {
    triggerKind: 'any' | 'manual';
    debouncerDelay: number;
    enabled: boolean;
  }

  export const DEFAULT_SETTINGS: ISettings = {
    triggerKind: 'any',
    debouncerDelay: 250,
    enabled: false
  };
}
