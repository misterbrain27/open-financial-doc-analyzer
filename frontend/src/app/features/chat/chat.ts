import { Component, ElementRef, effect, inject, signal, viewChild } from '@angular/core';
import { QueryService, Source } from '../services/query.service';

interface ChatMessage {
  role: 'user' | 'assistant';
  text: string;
  sources?: Source[];
  error?: string;
}

@Component({
  selector: 'app-chat',
  templateUrl: './chat.html',
  styleUrl: './chat.scss',
})
export class Chat {
  private _queryService = inject(QueryService);
  private _scroller = viewChild<ElementRef<HTMLElement>>('scroller');

  messages = signal<ChatMessage[]>([]);
  streamingText = signal('');
  streamingSources = signal<Source[] | null>(null);
  streaming = signal(false);

  /** Suggested starter prompts shown on the empty screen, to illustrate the kind of question expected. */
  readonly suggestions = [
    'Quel est le chiffre d’affaires de l’exercice ?',
    'Quels sont les principaux risques identifiés ?',
    'Comment a évolué la marge opérationnelle ?',
  ];

  constructor() {
    effect(() => {
      // Dependencies: anything that grows the conversation thread.
      this.messages();
      this.streamingText();

      const element = this._scroller()?.nativeElement;
      if (!element) {
        return;
      }
      // The effect runs before the DOM reflects the new content: we wait for the next
      // frame to measure the correct height.
      requestAnimationFrame(() => {
        element.scrollTop = element.scrollHeight;
      });
    });
  }

  ask(input: HTMLInputElement, query: string) {
    input.value = query;
    this.onSubmit(input);
  }

  onSubmit(input: HTMLInputElement) {
    const query = input.value.trim();
    if (!query || this.streaming()) {
      return;
    }
    input.value = '';

    this.messages.update((msgs) => [...msgs, { role: 'user', text: query }]);
    this.streamingText.set('');
    this.streamingSources.set(null);
    this.streaming.set(true);

    this._queryService.streamQuery({ query }).subscribe({
      next: (event) => {
        switch (event.type) {
          case 'sources':
            this.streamingSources.set(event.sources);
            break;
          case 'delta':
            this.streamingText.update((text) => text + event.text);
            break;
          case 'done':
            this._commitAssistantMessage();
            break;
          case 'error':
            this._commitAssistantMessage(event.message);
            break;
        }
      },
      error: () => {
        this._commitAssistantMessage('Une erreur est survenue.');
      },
    });
  }

  private _commitAssistantMessage(error?: string) {
    this.messages.update((msgs) => [
      ...msgs,
      {
        role: 'assistant',
        text: this.streamingText(),
        sources: this.streamingSources() ?? undefined,
        error,
      },
    ]);
    this.streamingText.set('');
    this.streamingSources.set(null);
    this.streaming.set(false);
  }
}
