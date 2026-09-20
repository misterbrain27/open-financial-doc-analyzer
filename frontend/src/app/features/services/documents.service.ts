import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';


 export interface DocumentResponse {
    id: number,
    source_path : string,
    company : string | null,
    year : number | null
 }

@Injectable({
  providedIn: 'root',
})
export class DocumentsService {

  private _http = inject(HttpClient);

  postFormData(form: FormData): Observable<DocumentResponse>
  {
     return this._http.post<DocumentResponse>('/ingest', form);
  }
  
}
