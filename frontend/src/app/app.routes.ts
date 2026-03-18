import { Routes } from '@angular/router';
import { PortalPage } from './components/portal-page/portal-page';
import { DashboardPage } from './components/dashboard-page/dashboard-page';
import { authGuard } from './guards/auth-guard';
import { VerifyAccountPage } from './components/verify-account-page/verify-account-page';
import { RegisterForm } from './components/portal-page/register-form/register-form';
import { LoginForm } from './components/portal-page/login-form/login-form';

export const routes: Routes = [
  { path: '', redirectTo: 'portal', pathMatch: 'full' },
  {
    path: 'portal',
    component: PortalPage,
    children: [
      { path: '', redirectTo: 'login', pathMatch: 'full' },
      { path: 'register', component: RegisterForm },
      { path: 'login', component: LoginForm },
    ],
  },
  { path: 'dashboard', component: DashboardPage, canActivate: [authGuard] }, // data: { role: 'admin' } for admins
  { path: 'verify-account', component: VerifyAccountPage },
];
