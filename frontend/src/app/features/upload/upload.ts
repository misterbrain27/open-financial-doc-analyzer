import { Component, computed, inject, signal } from '@angular/core';
import { HttpErrorResponse } from '@angular/common/http';
import { MatButtonModule } from '@angular/material/button';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { DocumentResponse, DocumentsService } from '../services/documents.service';

@Component({
  selector: 'app-upload',
  imports: [MatButtonModule, MatProgressBarModule],
  templateUrl: './upload.html',
  styleUrl: './upload.scss',
})
export class Upload {
  private documentService = inject(DocumentsService);

  selectedFile = signal<File | null>(null);
  uploading = signal<boolean>(false);
  result = signal<DocumentResponse | null>(null);
  error = signal<string | null>(null);

  /** Vrai tant qu'un fichier survole la zone de dépôt (surbrillance de la dropzone). */
  dragging = signal<boolean>(false);

  fileSize = computed(() => {
    const size = this.selectedFile()?.size;
    if (size === undefined) {
      return null;
    }
    const mb = size / 1024 / 1024;
    return mb >= 1 ? `${mb.toFixed(1)} Mo` : `${Math.max(1, Math.round(size / 1024))} Ko`;
  });

  onDragOver(event: DragEvent) {
    // Sans preventDefault, le navigateur refuse le drop et ouvrirait le PDF dans un onglet.
    event.preventDefault();
    this.dragging.set(true);
  }

  onDragLeave() {
    this.dragging.set(false);
  }

  onDrop(event: DragEvent) {
    event.preventDefault();
    this.dragging.set(false);
    this.select(event.dataTransfer?.files?.[0] ?? null);
  }

  select(file: File | null) {
    if (file && !file.name.toLowerCase().endsWith('.pdf')) {
      this.error.set('Seuls les fichiers PDF sont acceptés.');
      return;
    }
    this.error.set(null);
    this.result.set(null);
    this.selectedFile.set(file);
  }

  clear() {
    this.selectedFile.set(null);
    this.result.set(null);
    this.error.set(null);
  }

  onSubmit(file: File | null) {
    if (!file) {
      return;
    }

    this.uploading.set(true);
    this.error.set(null);

    const form = new FormData();
    form.append('file', file);
    this.documentService.postFormData(form).subscribe({
      next: (res: DocumentResponse) => {
        this.result.set(res);
      },
      error: (err: HttpErrorResponse) => {
        this.error.set(err.error?.detail ?? 'Une erreur est survenue.');
        this.result.set(null);
        this.uploading.set(false);
      },
      complete: () => {
        this.uploading.set(false);
      },
    });
  }
}
