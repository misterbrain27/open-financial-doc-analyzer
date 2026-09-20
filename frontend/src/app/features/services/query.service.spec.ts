import { TestBed } from '@angular/core/testing';
import { HttpEventType, provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { parseSseChunk, QueryEvent, QueryService } from './query.service';

describe('parseSseChunk', () => {
  it('parses a single complete frame', () => {
    const buffer = 'event: delta\ndata: {"text":"Bonjour"}\n\n';

    const { events, rest } = parseSseChunk(buffer);

    expect(events).toEqual([{ type: 'delta', text: 'Bonjour' }]);
    expect(rest).toBe('');
  });
  it('parses multiple complete frames', () => {
    const buffer = 'event: delta\ndata: {"text":"Bonjour"}\n\n' + 'event: done\ndata: {}\n\n';

    const { events, rest } = parseSseChunk(buffer);

    expect(events).toEqual([{ type: 'delta', text: 'Bonjour' }, { type: 'done' }]);
    expect(rest).toBe('');
  });

  it('handles a frame split across two chunks', () => {
    // JSON truncated mid-way, like a network chunk cutting a frame off before its end.
    const chunk1 = 'event: delta\ndata: {"text": "Bonj';
    const first = parseSseChunk(chunk1);

    // No "\n\n" in chunk1 => no complete frame: everything goes into `rest`, nothing to parse.
    expect(first.events).toEqual([]);
    expect(first.rest).toBe(chunk1);

    // Reproduces what streamQuery does: the previous round's `rest` is prepended to
    // the newly received text, before reparsing.
    const chunk2 = first.rest + 'our"}\n\n';
    const second = parseSseChunk(chunk2);

    // The frame is now complete and the JSON correctly reconstituted ("Bonjour", not
    // "Bonj" + "our" separately).
    expect(second.events).toEqual([{ type: 'delta', text: 'Bonjour' }]);
    expect(second.rest).toBe('');
  });

  it('ignores an unknown event type', () => {
    const buffer = 'event: ping\ndata: {}\n\n' + 'event: delta\ndata: {"text":"Bonjour"}\n\n';

    const { events, rest } = parseSseChunk(buffer);

    expect(events).toEqual([{ type: 'delta', text: 'Bonjour' }]);
    expect(rest).toBe('');
  });

  it('maps an error frame to a QueryEvent with the backend message', () => {
    const buffer = 'event: error\ndata: {"message":"Backend indisponible"}\n\n';

    const { events, rest } = parseSseChunk(buffer);

    expect(events).toEqual([{ type: 'error', message: 'Backend indisponible' }]);
    expect(rest).toBe('');
  });
});

describe('QueryService.streamQuery', () => {
  let service: QueryService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(QueryService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('POSTe le corps de la requête à /query', () => {
    service.streamQuery({ query: 'Quel est le CA ?' }).subscribe();

    const req = httpMock.expectOne('/query');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ query: 'Quel est le CA ?' });

    req.flush('');
  });

  it("parse les DownloadProgress cumulatifs en QueryEvents, dans l'ordre, puis complète", () => {
    const received: QueryEvent[] = [];
    let completed = false;

    service.streamQuery({ query: 'Quel est le CA ?' }).subscribe({
      next: (event) => received.push(event),
      complete: () => (completed = true),
    });

    const req = httpMock.expectOne('/query');

    // `partialText` is cumulative: each DownloadProgress event contains ALL the text
    // received so far, not just the new chunk (as HttpClient actually does).
    const frame1 = 'event: delta\ndata: {"text": "Bonjour"}\n\n';
    req.event({
      type: HttpEventType.DownloadProgress,
      loaded: frame1.length,
      partialText: frame1,
    } as any);

    const frame2 = frame1 + 'event: done\ndata: {}\n\n';
    req.event({
      type: HttpEventType.DownloadProgress,
      loaded: frame2.length,
      partialText: frame2,
    } as any);

    req.flush('');

    expect(received).toEqual([{ type: 'delta', text: 'Bonjour' }, { type: 'done' }]);
    expect(completed).toBe(true);
  });
});
