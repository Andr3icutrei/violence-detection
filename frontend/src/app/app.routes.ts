import { Routes } from '@angular/router';
import { LoginPageComponent } from './components/login-page-component/login-page';

export const routes: Routes = [
  { path: '', redirectTo: 'login', pathMatch: 'full' },
  { path: 'login', component: LoginPageComponent },
];
