import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { TranslatePipe } from '@ngx-translate/core';
import {
  GoogleSigninButtonModule,
  SocialAuthService,
  SocialUser,
} from '@abacritt/angularx-social-login';
import { Subscription } from 'rxjs';
import { RouterLink } from '@angular/router';
import { ControlError } from '../../control-error/control-error';

@Component({
  selector: 'app-login-form',
  imports: [
    ReactiveFormsModule,
    TranslatePipe,
    GoogleSigninButtonModule,
    RouterLink,
    ControlError
  ],
  standalone: true,
  templateUrl: './login-form.html',
  styleUrl: './login-form.css',
})
export class LoginForm implements OnInit, OnDestroy {
  form: FormGroup;
  user: SocialUser | null = null;
  loggedIn: boolean = false;
  isSubmitted: boolean = false;
  private authSubscription!: Subscription;

  constructor(
    private fb: FormBuilder,
    private authService: SocialAuthService,
  ) {
    this.form = this.fb.group({
      email: ['', [Validators.required, Validators.email, Validators.maxLength(50)]],
      password: ['', [Validators.required, Validators.maxLength(50)]],
      rememberMe: [false],
    });
  }

  ngOnInit() {
    this.authSubscription = this.authService.authState.subscribe((user) => {
      this.user = user;
      this.loggedIn = user != null;

      if (this.loggedIn) {
        console.log('Datele utilizatorului:', this.user);
        // Aici extragi this.user.idToken și faci request-ul HTTP către backend
      }
    });
  }

  isControlRequired(controlName: string): boolean {
    if(this.form.contains(controlName)) {
      const control = this.form.get(controlName);
      return control?.hasValidator(Validators.required) || false;
    }
    return false;
  }

  isControlValid(controlName: string): boolean {
    if (this.form.contains(controlName)) {
      const control = this.form.get(controlName);
      return control!.dirty && control!.invalid;
    }
    return false;
  }

  isFormValid(): boolean {
    return this.form.valid && !this.isSubmitted;
  }

  onSubmit(): void {

  }

  ngOnDestroy() {
    if (this.authSubscription) {
      this.authSubscription.unsubscribe();
    }
  }
}
