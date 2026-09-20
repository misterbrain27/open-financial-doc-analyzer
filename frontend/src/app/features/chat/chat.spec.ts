import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Subject } from 'rxjs';

import { Chat } from './chat';
import { QueryEvent, QueryService, Source } from '../services/query.service';

function inputWithValue(value: string): HTMLInputElement {
  const input = document.createElement('input');
  input.value = value;
  return input;
}

describe('Chat', () => {
  let component: Chat;
  let fixture: ComponentFixture<Chat>;
  let queryServiceSpy: { streamQuery: ReturnType<typeof vi.fn> };
  let events$: Subject<QueryEvent>;

  beforeEach(async () => {
    events$ = new Subject<QueryEvent>();
    queryServiceSpy = { streamQuery: vi.fn(() => events$.asObservable()) };

    await TestBed.configureTestingModule({
      imports: [Chat],
      providers: [{ provide: QueryService, useValue: queryServiceSpy }],
    }).compileComponents();

    fixture = TestBed.createComponent(Chat);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it("ignore une soumission vide et n'appelle pas le service", () => {
    component.onSubmit(inputWithValue('   '));

    expect(queryServiceSpy.streamQuery).not.toHaveBeenCalled();
    expect(component.messages()).toEqual([]);
  });

  it("pousse le message utilisateur, vide l'input et démarre le streaming", () => {
    const input = inputWithValue('Quel est le CA ?');

    component.onSubmit(input);

    expect(component.messages()).toEqual([{ role: 'user', text: 'Quel est le CA ?' }]);
    expect(input.value).toBe('');
    expect(component.streaming()).toBe(true);
    expect(queryServiceSpy.streamQuery).toHaveBeenCalledWith({ query: 'Quel est le CA ?' });
  });

  it('ignore une deuxième soumission pendant que le streaming est en cours', () => {
    component.onSubmit(inputWithValue('Question 1'));
    component.onSubmit(inputWithValue('Question 2'));

    expect(queryServiceSpy.streamQuery).toHaveBeenCalledTimes(1);
    expect(component.messages()).toEqual([{ role: 'user', text: 'Question 1' }]);
  });

  it('accumule sources et delta pendant le streaming, puis committe le message assistant sur done', () => {
    component.onSubmit(inputWithValue('Quel est le CA ?'));

    const sources: Source[] = [
      { index: 1, company: 'Acme', year: 2024, page_number: 3, text: '...' },
    ];
    events$.next({ type: 'sources', sources });
    events$.next({ type: 'delta', text: 'Le ' });
    events$.next({ type: 'delta', text: 'CA est de 100.' });

    // Still streaming: nothing is committed to messages() yet.
    expect(component.streamingSources()).toEqual(sources);
    expect(component.streamingText()).toBe('Le CA est de 100.');
    expect(component.messages().length).toBe(1);

    events$.next({ type: 'done' });

    expect(component.streaming()).toBe(false);
    expect(component.streamingText()).toBe('');
    expect(component.streamingSources()).toBeNull();
    expect(component.messages()).toEqual([
      { role: 'user', text: 'Quel est le CA ?' },
      { role: 'assistant', text: 'Le CA est de 100.', sources },
    ]);
  });

  it("committe un message d'erreur quand le service émet un évènement error", () => {
    component.onSubmit(inputWithValue('Quel est le CA ?'));

    events$.next({ type: 'error', message: 'Backend indisponible' });

    const last = component.messages().at(-1);
    expect(last?.role).toBe('assistant');
    expect(last?.error).toBe('Backend indisponible');
    expect(component.streaming()).toBe(false);
  });

  it("committe un message d'erreur générique si l'observable lui-même échoue", () => {
    component.onSubmit(inputWithValue('Quel est le CA ?'));

    events$.error(new Error('network down'));

    const last = component.messages().at(-1);
    expect(last?.error).toBe('Une erreur est survenue.');
    expect(component.streaming()).toBe(false);
  });
});
