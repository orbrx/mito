import { ActionEnum, KeyboardShorcut } from "../types"
import { Actions } from "./actions"

/**
 * A list of all keyboard shortcuts in the app.
 */
export const keyboardShortcuts: KeyboardShorcut[] = [
    {
        macKeyCombo: {
            metaKey: true,
            keys: ['c']
        },
        winKeyCombo: {
            ctrlKey: true,
            keys: ['c']
        },
        action: ActionEnum.Copy
    },
    {
        macKeyCombo: {
            metaKey: true,
            keys: ['f']
        },
        winKeyCombo: {
            ctrlKey: true,
            keys: ['f']
        },
        preventDefaultAndStopPropagation: true,
        action: ActionEnum.OpenFind,
        jupyterLabCommand: 'mitosheet:open-search',
    },
    {
        macKeyCombo: {
            metaKey: true,
            keys: ['z']
        },
        winKeyCombo: {
            ctrlKey: true,
            keys: ['z']
        },
        preventDefaultAndStopPropagation: true,
        jupyterLabCommand: 'mitosheet:mito-undo',
        action: ActionEnum.Undo
    },
    {
        macKeyCombo: {
            metaKey: true,
            keys: ['y']
        },
        winKeyCombo: {
            ctrlKey: true,
            keys: ['y']
        },
        preventDefaultAndStopPropagation: true,
        action: ActionEnum.Redo
    },
    {
        macKeyCombo: {
            altKey: true,
            keys: ['ArrowLeft']
        },
        winKeyCombo: {
            altKey: true,
            // TODO: this  is not consistent with the Excel shortcut in windows. 
            // But Edge grabs focus when pressing ctrl+PageUp/PageDown (which is the Excel shortcut)
            keys: ['ArrowLeft']
        },
        action: ActionEnum.Open_Previous_Sheet
    },
    {
        macKeyCombo: {
            altKey: true,
            keys: ['ArrowRight']
        },
        winKeyCombo: {
            altKey: true,
            // TODO: this  is not consistent with the Excel shortcut in windows. 
            // But Edge grabs focus when pressing ctrl+PageUp/PageDown (which is the Excel shortcut)
            keys: ['ArrowRight']
        },
        action: ActionEnum.Open_Next_Sheet
    },
    {
        macKeyCombo: {
            ctrlKey: true,
            shiftKey: true,
            keys: ['F']
        },
        winKeyCombo: {
            ctrlKey: true,
            keys: ['h']
        },
        action: ActionEnum.OpenFindAndReplace
    }
]

/**
 * Handles keyboard shortcuts. If a keyboard shortcut is pressed, the corresponding action is executed.
 * @param e The keyboard event.
 * @param actions The actions object.
 */
export const handleKeyboardShortcuts = (e: React.KeyboardEvent, actions: Actions) => {
    const operatingSystem = window.navigator.userAgent.toUpperCase().includes('MAC')
        ? 'mac'
        : 'windows'

    const shortcut = keyboardShortcuts.find(shortcut => {
        const keyCombo = operatingSystem === 'mac' ? shortcut.macKeyCombo : shortcut.winKeyCombo
        // If the special key combination doesn't match, return false.
        if (!!e.metaKey !== !!keyCombo.metaKey ||
            !!e.ctrlKey !== !!keyCombo.ctrlKey ||
            !!e.shiftKey !== !!keyCombo.shiftKey ||
            !!e.altKey !== !!keyCombo.altKey) {
            return false;
        }
        // If the special keys matched, check if the key is the same.
        return keyCombo.keys.includes(e.key);
    })
    
    if (shortcut !== undefined && !actions.buildTimeActions[shortcut.action].isDisabled()) {
        actions.buildTimeActions[shortcut.action].actionFunction()
        if (shortcut.preventDefaultAndStopPropagation) {
            e.preventDefault();
            e.stopPropagation();
        }
    }
}