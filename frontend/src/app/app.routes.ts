import { Routes } from '@angular/router';
import { PortalPage } from './components/portal-page/portal-page';
import { DashboardPage } from './components/dashboard-page/dashboard-page';
import { authGuard } from './guards/auth-guard';
import { VerifyAccountPage } from './components/verify-account-page/verify-account-page';
import { RegisterForm } from './components/portal-page/register-form/register-form';
import { LoginForm } from './components/portal-page/login-form/login-form';
import { ForgotPasswordForm } from './components/portal-page/forgot-password-form/forgot-password-form';
import {
  VerifyAccountUnexistentUser
} from './components/redirect-widgets/verify-account-unexistent-user/verify-account-unexistent-user';
import {
  VerifyAccountAlreadyVerified
} from './components/redirect-widgets/verify-account-already-verified/verify-account-already-verified';
import {
  VerifyAccountInvalidToken
} from './components/redirect-widgets/verify-account-invalid-token/verify-account-invalid-token';
import { VerifyAccountSuccess } from './components/redirect-widgets/verify-account-success/verify-account-success';
import {
  VerifyAccountServerError
} from './components/redirect-widgets/verify-account-server-error/verify-account-server-error';
import { ResetPasswordPage } from './components/reset-password-page/reset-password-page';
import { ResetPasswordForm } from './components/redirect-widgets/reset-password-form/reset-password-form';
import {
  ResetPasswordInvalidToken
} from './components/redirect-widgets/reset-password-invalid-token/reset-password-invalid-token';

export const routes: Routes = [
  { path: '', redirectTo: 'portal', pathMatch: 'full' },
  {
    path: 'portal',
    component: PortalPage,
    children: [
      { path: '', redirectTo: 'login', pathMatch: 'full' },
      { path: 'register', component: RegisterForm },
      { path: 'login', component: LoginForm },
      { path: 'reset-password', component: ForgotPasswordForm }
    ],
  },
  { path: 'dashboard', component: DashboardPage, canActivate: [authGuard] }, // data: { role: 'admin' } for admins
  {
    path: 'verify-account',
    component: VerifyAccountPage,
    children: [
      { path: 'already-verified', component: VerifyAccountAlreadyVerified },
      { path: 'invalid-token', component: VerifyAccountInvalidToken },
      { path: 'server-error', component: VerifyAccountServerError },
      { path: 'success', component: VerifyAccountSuccess },
      { path: 'unexistent-user', component: VerifyAccountUnexistentUser },
    ]
  },
  {
    path: 'reset-password',
    component: ResetPasswordPage,
    children: [
      { path: 'reset', component: ResetPasswordForm },
      { path: 'invalid-token', component: ResetPasswordInvalidToken },
    ]
  }
];
