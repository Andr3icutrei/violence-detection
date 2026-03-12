import { Routes } from '@angular/router';
import { LoginPage } from './components/login-page/login-page';

export const routes: Routes = [
  { path: '', redirectTo: 'login', pathMatch: 'full' },
  { path: 'login', component: LoginPage },
];
