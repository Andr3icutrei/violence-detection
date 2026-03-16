import { Routes } from '@angular/router';
import { PortalPage } from './components/portal-page/portal-page';
import { DashboardPage } from './components/dashboard-page/dashboard-page';
import { authGuard } from './guards/auth-guard';

export const routes: Routes = [
  { path: '', redirectTo: 'portal', pathMatch: 'full' },
  { path: 'portal', component: PortalPage },
  { path: 'dashboard', component: DashboardPage, canActivate: [authGuard] }, // data: { role: 'admin' } for admins
];
