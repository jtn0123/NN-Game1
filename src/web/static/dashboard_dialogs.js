// ============================================================
// DASHBOARD DIALOG HELPERS
// ============================================================

(function(global) {
    const FOCUSABLE_SELECTOR = [
        'a[href]',
        'button:not([disabled])',
        'textarea:not([disabled])',
        'input:not([disabled])',
        'select:not([disabled])',
        '[tabindex]:not([tabindex="-1"])',
    ].join(',');

    function removeDialog(backdrop) {
        if (backdrop && backdrop.parentNode) {
            backdrop.parentNode.removeChild(backdrop);
        }
    }

    function focusableElements(container) {
        if (!container || typeof container.querySelectorAll !== 'function') {
            return [];
        }
        return Array.from(container.querySelectorAll(FOCUSABLE_SELECTOR)).filter((element) => (
            !element.disabled && element.getAttribute?.('aria-hidden') !== 'true'
        ));
    }

    function restoreFocus(element) {
        if (element && typeof element.focus === 'function') {
            element.focus();
        }
    }

    function appendParagraph(parent, text) {
        if (!text) {
            return;
        }
        const paragraph = document.createElement('p');
        paragraph.className = 'action-dialog-message';
        paragraph.textContent = text;
        parent.appendChild(paragraph);
    }

    function appendDetails(parent, details) {
        if (!Array.isArray(details) || details.length === 0) {
            return;
        }
        const list = document.createElement('ul');
        list.className = 'action-dialog-details';
        details.forEach((detail) => {
            const item = document.createElement('li');
            item.textContent = detail;
            list.appendChild(item);
        });
        parent.appendChild(list);
    }

    function buttonForChoice(choice) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `dialog-btn ${choice.variant || 'secondary'}`.trim();
        button.textContent = choice.label;
        button.dataset.choice = choice.value ?? '';
        return button;
    }

    function openDialog(options) {
        return new Promise((resolve) => {
            if (typeof document === 'undefined') {
                resolve(options.defaultValue ?? null);
                return;
            }

            const previouslyFocused = document.activeElement;
            const backdrop = document.createElement('div');
            backdrop.className = 'action-dialog-backdrop';

            const dialog = document.createElement('section');
            dialog.className = 'action-dialog';
            dialog.tabIndex = -1;
            dialog.setAttribute('role', 'dialog');
            dialog.setAttribute('aria-modal', 'true');
            dialog.setAttribute('aria-labelledby', 'action-dialog-title');

            const title = document.createElement('h3');
            title.id = 'action-dialog-title';
            title.textContent = options.title || 'Confirm action';
            dialog.appendChild(title);

            appendParagraph(dialog, options.message);
            appendDetails(dialog, options.details);

            const footer = document.createElement('div');
            footer.className = 'action-dialog-footer';

            const cancelChoice = {
                value: options.cancelValue ?? null,
                label: options.cancelText || 'Cancel',
                variant: 'secondary',
            };
            const choices = [...(options.choices || []), cancelChoice];
            choices.forEach((choice) => footer.appendChild(buttonForChoice(choice)));

            dialog.appendChild(footer);
            backdrop.appendChild(dialog);
            document.body.appendChild(backdrop);

            const finish = (value) => {
                document.removeEventListener('keydown', onKeyDown);
                removeDialog(backdrop);
                restoreFocus(previouslyFocused);
                resolve(value);
            };

            function trapFocus(event) {
                const focusable = focusableElements(dialog);
                if (focusable.length === 0) {
                    event.preventDefault();
                    dialog.focus();
                    return;
                }

                const first = focusable[0];
                const last = focusable[focusable.length - 1];
                const activeElement = document.activeElement;
                if (event.shiftKey && (activeElement === first || !dialog.contains(activeElement))) {
                    event.preventDefault();
                    last.focus();
                    return;
                }
                if (!event.shiftKey && activeElement === last) {
                    event.preventDefault();
                    first.focus();
                }
            }

            function onKeyDown(event) {
                if (event.key === 'Escape') {
                    finish(cancelChoice.value);
                    return;
                }
                if (event.key === 'Tab') {
                    trapFocus(event);
                }
            }

            footer.addEventListener('click', (event) => {
                const button = event.target.closest('button[data-choice]');
                if (!button) {
                    return;
                }
                finish(button.dataset.choice || null);
            });
            backdrop.addEventListener('click', (event) => {
                if (event.target === backdrop) {
                    finish(cancelChoice.value);
                }
            });
            document.addEventListener('keydown', onKeyDown);

            const preferredButton = footer.querySelector('.primary, .danger, button')
                || focusableElements(dialog)[0];
            if (preferredButton && typeof preferredButton.focus === 'function') {
                preferredButton.focus();
            } else {
                dialog.focus();
            }
        });
    }

    function ask(options) {
        return openDialog({
            ...options,
            choices: [
                {
                    value: 'yes',
                    label: options.confirmText || 'Continue',
                    variant: options.danger ? 'danger' : 'primary',
                },
            ],
        }).then((value) => value === 'yes');
    }

    function choose(options) {
        return openDialog(options);
    }

    global.DashboardDialogs = Object.freeze({ ask, choose });
})(typeof window !== 'undefined' ? window : globalThis);
