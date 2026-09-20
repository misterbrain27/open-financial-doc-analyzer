import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';

import { DocumentResponse, DocumentsService } from './documents.service';

describe('DocumentsService', () => {
  let service: DocumentsService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(DocumentsService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('POSTe le FormData à /ingest et renvoie le document parsé', () => {
    const form = new FormData();
    form.append('file', new File(['dummy'], 'rapport.pdf'));
    const response: DocumentResponse = {
      id: 1,
      source_path: 'rapport.pdf',
      company: 'Acme',
      year: 2024,
    };
    let received: DocumentResponse | undefined;

    service.postFormData(form).subscribe((res) => (received = res));

    const req = httpMock.expectOne('/ingest');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toBe(form);

    req.flush(response);

    expect(received).toEqual(response);
  });
});
