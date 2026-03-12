import { Component, OnInit } from '@angular/core';
import { TranslatePipe } from '@ngx-translate/core';
import { FormBuilder, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { RouterOutlet } from '@angular/router';
import { LoginForm } from './login-form/login-form';

@Component({
  selector: 'app-login-page',
  imports: [
    TranslatePipe,
    ReactiveFormsModule,
    LoginForm
  ],
  standalone: true,
  templateUrl: './login-page.html',
  styleUrl: './login-page.css',
})
export class LoginPage {

}
