import type { Page } from '@playwright/test';

// Centralized DTOs for E2E: we redefine here the shape of data exchanged with the API,
// instead of importing from src/ (keeps E2E decoupled from the Angular code). This is the
// ONLY place to update if the backend contract changes.
export interface Source {
  index: number;
  company: string | null;
  year: number | null;
  page_number: number;
  text: string;
}

export interface IngestedDocument {
  id: number;
  source_path: string;
  company: string | null;
  year: number | null;
}

// Serializes a frame in the EXACT SSE format emitted by the backend (`_sse_event`):
//   event: <type>\ndata: <json>\n\n
// We go through JSON.stringify for the `data`: zero escaping worries (French apostrophes,
// quotes…), which we couldn't guarantee by writing the JSON by hand.
function sseFrame(type: string, data: unknown): string {
  return `event: ${type}\ndata: ${JSON.stringify(data)}\n\n`;
}

// Responds to POST /query with an SSE stream. The frontend reads the response as cumulative
// text (XHR reportProgress): returning the whole body at once is enough, `parseSseChunk`
// splits the frames itself. So we never need to simulate real network chunking.
async function fulfillSse(page: Page, body: string): Promise<void> {
  await page.route('**/query', async (route) => {
    await route.fulfill({ status: 200, contentType: 'text/event-stream', body });
  });
}

export interface MockQueryOptions {
  answer: string;
  sources: Source[];
}

// Nominal case: sources → response (a single delta) → done.
export async function mockQuery(page: Page, { answer, sources }: MockQueryOptions): Promise<void> {
  const body =
    sseFrame('sources', { sources }) + sseFrame('delta', { text: answer }) + sseFrame('done', {});
  await fulfillSse(page, body);
}

export interface MockQueryErrorOptions {
  message: string;
  partialAnswer?: string;
  sources?: Source[];
}

// Error mid-stream: sources → (start of response) → error (no `done`).
// Reproduces a generation-side failure AFTER retrieval has already returned its passages.
export async function mockQueryError(
  page: Page,
  { message, partialAnswer = '', sources = [] }: MockQueryErrorOptions,
): Promise<void> {
  let body = sseFrame('sources', { sources });
  if (partialAnswer) {
    body += sseFrame('delta', { text: partialAnswer });
  }
  body += sseFrame('error', { message });
  await fulfillSse(page, body);
}

// POST /ingest — plain JSON response (much simpler than SSE).
export async function mockIngest(page: Page, document: IngestedDocument): Promise<void> {
  await page.route('**/ingest', async (route) => {
    await route.fulfill({ json: document });
  });
}

export interface MockIngestErrorOptions {
  status?: number;
  detail: string;
}

// POST /ingest failure: HTTP status >= 400 + `{ detail }` body, which the component reads via
// `err.error.detail` to display the error message.
export async function mockIngestError(
  page: Page,
  { status = 400, detail }: MockIngestErrorOptions,
): Promise<void> {
  await page.route('**/ingest', async (route) => {
    await route.fulfill({ status, json: { detail } });
  });
}
