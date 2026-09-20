import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpErrorResponse } from '@angular/common/http';
import { Subject, of, throwError } from 'rxjs';

import { Upload } from './upload';
import { DocumentResponse, DocumentsService } from '../services/documents.service';

describe('Upload', () => {
  let component: Upload;
  let fixture: ComponentFixture<Upload>;
  let documentsServiceSpy: { postFormData: ReturnType<typeof vi.fn> };

  beforeEach(async () => {
    documentsServiceSpy = { postFormData: vi.fn() };

    await TestBed.configureTestingModule({
      imports: [Upload],
      providers: [{ provide: DocumentsService, useValue: documentsServiceSpy }],
    }).compileComponents();

    fixture = TestBed.createComponent(Upload);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it("n'appelle pas le service si aucun fichier n'est sélectionné", () => {
    component.onSubmit(null);

    expect(documentsServiceSpy.postFormData).not.toHaveBeenCalled();
  });

  it('passe uploading à true avant même que la réponse arrive', () => {
    const subject = new Subject<DocumentResponse>();
    documentsServiceSpy.postFormData.mockReturnValue(subject.asObservable());
    const file = new File(['dummy'], 'rapport.pdf', { type: 'application/pdf' });

    component.onSubmit(file);

    // The request hasn't responded yet (the Subject hasn't emitted anything): uploading must
    // already be set to true synchronously, before the HTTP call, not in the success callback.
    expect(component.uploading()).toBe(true);

    subject.next({ id: 1, source_path: 'rapport.pdf', company: null, year: null });
    subject.complete();

    expect(component.uploading()).toBe(false);
    expect(component.result()).toEqual({
      id: 1,
      source_path: 'rapport.pdf',
      company: null,
      year: null,
    });
  });

  it('stocke le résultat sur succès', () => {
    const response: DocumentResponse = {
      id: 1,
      source_path: 'rapport.pdf',
      company: 'Acme',
      year: 2024,
    };
    documentsServiceSpy.postFormData.mockReturnValue(of(response));
    const file = new File(['dummy'], 'rapport.pdf', { type: 'application/pdf' });

    component.onSubmit(file);

    expect(component.result()).toEqual(response);
    expect(component.error()).toBeNull();
    expect(component.uploading()).toBe(false);
  });

  it('extrait err.error.detail et réinitialise le résultat sur échec', () => {
    const httpError = new HttpErrorResponse({
      error: { detail: 'Fichier invalide' },
      status: 400,
    });
    documentsServiceSpy.postFormData.mockReturnValue(throwError(() => httpError));
    const file = new File(['dummy'], 'rapport.pdf', { type: 'application/pdf' });

    component.onSubmit(file);

    expect(component.error()).toBe('Fichier invalide');
    expect(component.result()).toBeNull();
    expect(component.uploading()).toBe(false);
  });

  it('retombe sur un message générique si err.error.detail est absent', () => {
    const httpError = new HttpErrorResponse({ status: 500 });
    documentsServiceSpy.postFormData.mockReturnValue(throwError(() => httpError));
    const file = new File(['dummy'], 'rapport.pdf', { type: 'application/pdf' });

    component.onSubmit(file);

    expect(component.error()).toBe('Une erreur est survenue.');
  });
});
