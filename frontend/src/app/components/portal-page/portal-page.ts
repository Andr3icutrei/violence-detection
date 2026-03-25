import { Component } from '@angular/core';
import { LoginForm } from './login-form/login-form';
import { ForgotPasswordForm } from './forgot-password-form/forgot-password-form';
import { RegisterForm } from './register-form/register-form';
import { PortalForm } from './portal-form.type';
import { FormDescription } from './form-description/form-description';
import { RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-portal-page',
  imports: [
    RouterOutlet,
  ],
  standalone: true,
  templateUrl: './portal-page.html',
  styleUrl: './portal-page.css',
})
export class PortalPage {
}
