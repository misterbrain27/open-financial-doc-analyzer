import { HttpClient, HttpDownloadProgressEvent, HttpEventType } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

export interface QueryRequestBody {
  query: string;
  k?: number;
  company?: string | null;
  year?: number | null;
}

export interface Source {
  index: number;
  company: string | null;
  year: number | null;
  page_number: number;
  text: string;
}

export type QueryEvent =
  | { type: 'sources'; sources: Source[] }
  | { type: 'delta'; text: string }
  | { type: 'done' }
  | { type: 'error'; message: string };

// TODO: implement this yourself.
//
// `buffer` contains everything received so far that hasn't been turned into a QueryEvent yet.
// The frame format is exactly the one emitted by `_sse_event`
// (backend/app/api/routes.py):
//
//   event: <type>\ndata: <json>\n\n
//
// A complete frame always ends with a blank line ("\n\n"). But a network chunk can cut a
// frame off in the middle (nothing guarantees it arrives all at once) — that's why the
// function also returns `rest`: the remaining text (an incomplete frame, or an empty
// string), to be prepended to the next chunk received.
//
// - split `buffer` on "\n\n" to separate complete frames from the incomplete remainder
// - for each complete frame: grab the "event: xxx" line (the type) and the
//   "data: {...}" line (JSON.parse the content after "data: ")
// - build the corresponding QueryEvent based on the type ("sources" → { type: 'sources',
//   sources: data.sources }, "delta" → { type: 'delta', text: data.text }, "done" →
//   { type: 'done' }, "error" → { type: 'error', message: data.message })
export function parseSseChunk(buffer: string): { events: QueryEvent[]; rest: string } {
  const frames = buffer.split('\n\n');
  // The last element is either '' (buffer ending exactly on "\n\n", so everything is
  // complete), or an incomplete frame cut off in the middle — either way, that's what
  // needs to be kept for next time, not parsed now.
  const rest = frames.pop() ?? '';

  const events: QueryEvent[] = [];
  for (const frame of frames) {
    if (!frame.trim()) {
      continue;
    }

    const lines = frame.split('\n');
    const eventLine = lines.find((line) => line.startsWith('event: '));
    const dataLine = lines.find((line) => line.startsWith('data: '));
    if (!eventLine || !dataLine) {
      continue;
    }

    const type = eventLine.slice('event: '.length);
    const data = JSON.parse(dataLine.slice('data: '.length));

    switch (type) {
      case 'sources':
        events.push({ type: 'sources', sources: data.sources });
        break;
      case 'delta':
        events.push({ type: 'delta', text: data.text });
        break;
      case 'done':
        events.push({ type: 'done' });
        break;
      case 'error':
        events.push({ type: 'error', message: data.message });
        break;
    }
  }

  return { events, rest };
}

@Injectable({
  providedIn: 'root',
})
export class QueryService {
  private _http = inject(HttpClient);

  streamQuery(body: QueryRequestBody): Observable<QueryEvent> {
    return new Observable<QueryEvent>((subscriber) => {
      let consumed = 0;
      let buffer = '';

      const subscription = this._http
        .post('/query', body, {
          responseType: 'text',
          observe: 'events',
          reportProgress: true,
        })
        .subscribe({
          next: (event) => {
            if (event.type === HttpEventType.DownloadProgress) {
              const partialText = (event as HttpDownloadProgressEvent).partialText ?? '';
              buffer += partialText.slice(consumed);
              consumed = partialText.length;

              const { events, rest } = parseSseChunk(buffer);
              buffer = rest;
              for (const queryEvent of events) {
                subscriber.next(queryEvent);
              }
            } else if (event.type === HttpEventType.Response) {
              subscriber.complete();
            }
          },
          error: (err) => subscriber.error(err),
        });

      return () => subscription.unsubscribe();
    });
  }
}
