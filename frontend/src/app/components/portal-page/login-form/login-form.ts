import { Component, EventEmitter, OnDestroy, OnInit, Output } from '@angular/core';
import { FormBuilder, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { TranslatePipe } from '@ngx-translate/core';
import {
  GoogleSigninButtonModule,
  SocialAuthService,
  SocialUser,
} from '@abacritt/angularx-social-login';
import { Subscription } from 'rxjs';
import { ControlError } from '../../control-error/control-error';
import { FormDescription } from '../form-description/form-description';
import { PortalForm } from '../portal-form.type';
import { AuthService } from '../../../services/auth/auth-service';
import { HttpErrorResponse } from '@angular/common/http';
import { LoginRequestDto } from '../../../core/api/models/login-request-dto';
import { Router } from '@angular/router';
import { UserResponseDto } from '../../../core/api/models/user-response-dto';
import { FormSubmitDetail } from '../../form-submit-detail/form-submit-detail';

@Component({
  selector: 'app-login-form',
  imports: [
    ReactiveFormsModule,
    TranslatePipe,
    GoogleSigninButtonModule,
    ControlError,
    FormSubmitDetail,
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
  isPasswordVisible: boolean = false;

  @Output() formChange = new EventEmitter<PortalForm>();

  private authSubscription!: Subscription;

  constructor(
    private fb: FormBuilder,
    private socialAuthService: SocialAuthService,
    private authService: AuthService,
    private router: Router,
  ) {
    this.form = this.fb.group({
      email: ['', [Validators.required, Validators.email, Validators.maxLength(50)]],
      password: ['', [Validators.required, Validators.maxLength(50)]],
      rememberMe: [false],
    });
  }

  ngOnInit() {
    this.form.valueChanges.subscribe(() => {
      if (
        this.form.hasError('invalidCredentials') ||
        this.form.hasError('accessForbidden') ||
        this.form.hasError('serverError')
      ) {
        this.form.setErrors(null);
        this.form.updateValueAndValidity({ emitEvent: false });
      }
    });

    this.authSubscription = this.socialAuthService.authState.subscribe((user) => {
      this.user = user;
      this.loggedIn = user != null;

      if (this.loggedIn) {
        console.log('Datele utilizatorului:', this.user);
        // Aici extragi this.user.idToken și faci request-ul HTTP către backend
      }
    });
  }

  isControlRequired(controlName: string): boolean {
    if (this.form.contains(controlName)) {
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
    this.isSubmitted = true;
    this.authService.login(this.form.value.email, this.form.value.password).subscribe({
      next: (data: UserResponseDto) => {
        this.router.navigate(['dashboard']);
      },
      error: (err: HttpErrorResponse) => {
        console.log(err.status);
        if (err.status === 401) {
          this.form.setErrors({ invalidCredentials: true });
        }
        else if (err.status === 403) {
          this.form.setErrors({ accountNotVerified: true });
        } else if (err.status === 500) {
          this.form.setErrors({ serverError: true });
        }
        setTimeout(() => {
          this.isSubmitted = false;
          this.form.setErrors(null);
          this.form.updateValueAndValidity({ emitEvent: false });
        }, 5000);
      },
    });
  }

  togglePasswordVisibility(): void {
    this.isPasswordVisible = !this.isPasswordVisible;
  }

  switchForm(form: PortalForm): void {
    this.formChange.emit(form);
  }

  ngOnDestroy() {
    if (this.authSubscription) {
      this.authSubscription.unsubscribe();
    }
  }
}
