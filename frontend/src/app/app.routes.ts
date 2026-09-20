import { Routes } from '@angular/router';
import { Chat } from './features/chat/chat';
import { Upload } from './features/upload/upload';

export const routes: Routes = [
  { path: '', component: Upload },
  { path: 'chat', component: Chat },
];
