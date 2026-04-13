import {
  ChangeDetectorRef,
  Component,
  EventEmitter,
  OnDestroy,
  OnInit,
  Output,
} from '@angular/core';
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
import { Router, RouterLink } from '@angular/router';
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
    RouterLink,
    FormDescription,
  ],
  standalone: true,
  templateUrl: './login-form.html',
  styleUrl: './login-form.css',
})
export class LoginForm implements OnInit, OnDestroy {
  form: FormGroup;
  isSubmitted: boolean = false;
  isPasswordVisible: boolean = false;

  @Output() formChange = new EventEmitter<PortalForm>();

  private authSubscription!: Subscription;

  constructor(
    private fb: FormBuilder,
    private socialAuthService: SocialAuthService,
    private authService: AuthService,
    private router: Router,
    private cdr: ChangeDetectorRef,
  ) {
    this.form = this.fb.group({
      email: ['', [Validators.required, Validators.email, Validators.maxLength(50)]],
      password: ['', [Validators.required, Validators.maxLength(50)]],
    });
  }

  ngOnInit() {
    this.form.valueChanges.subscribe(() => {
      if (
        this.form.dirty &&
        (this.form.hasError('invalidCredentials') ||
          this.form.hasError('accountNotVerified') ||
          this.form.hasError('serverError') ||
          this.form.hasError('userNotFound'))
      ) {
        this.form.setErrors(null);
        this.form.updateValueAndValidity({ emitEvent: false });
      }
    });

    this.authSubscription = this.socialAuthService.authState.subscribe((user) => {
      if (user && user.idToken) {
        this.authService.loginWithGoogle(user.idToken).subscribe({
          next: () => {
            this.router.navigate(['dashboard']);
          },
          error: (err: HttpErrorResponse) => {
            if (err.status === 403) {
              this.form.setErrors({ accountNotVerified: true });
            } else if (err.status === 404) {
              this.form.setErrors({ userNotFound: true });
            } else if (err.status === 500) {
              this.form.setErrors({ serverError: true });
            } else {
              this.form.setErrors({ invalidCredentials: true });
            }
            this.cdr.detectChanges();
            setTimeout(() => {
              this.isSubmitted = false;
              this.form.setErrors(null);
              this.form.updateValueAndValidity({ emitEvent: false });
              this.cdr.detectChanges();
            }, 5000);
          },
        });
      }
    });
  }

  public isControlRequired(controlName: string): boolean {
    if (this.form.contains(controlName)) {
      const control = this.form.get(controlName);
      return control?.hasValidator(Validators.required) || false;
    }
    return false;
  }

  public isFormValid(): boolean {
    return this.form.valid && !this.isSubmitted;
  }

  public togglePasswordVisibility(): void {
    this.isPasswordVisible = !this.isPasswordVisible;
  }

  public switchForm(form: PortalForm): void {
    this.formChange.emit(form);
  }

  public onSubmit(): void {
    const email: string = this.form.value.email!;
    const password: string = this.form.value.password!;
    if (!email || !password) {
      return;
    }
    this.isSubmitted = true;
    this.form.markAsPristine();

    this.authService.login(email, password).subscribe({
      next: (data: UserResponseDto) => {
        this.router.navigate(['dashboard']);
        setTimeout(() => {
          this.isSubmitted = false;
          this.form.setErrors(null);
          this.form.updateValueAndValidity({ emitEvent: false });
          this.cdr.detectChanges();
        }, 5000);
      },
      error: (err: HttpErrorResponse) => {
        if (err.status === 401) {
          this.form.setErrors({ invalidCredentials: true });
        } else if (err.status === 403) {
          this.form.setErrors({ accountNotVerified: true });
        } else if (err.status === 404) {
          this.form.setErrors({ userNotFound: true });
        } else if (err.status === 500) {
          this.form.setErrors({ serverError: true });
        }
        this.cdr.detectChanges();
        setTimeout(() => {
          this.isSubmitted = false;
          this.form.setErrors(null);
          this.form.updateValueAndValidity({ emitEvent: false });
          this.cdr.detectChanges();
        }, 5000);
      },
    });
  }

  ngOnDestroy() {
    if (this.authSubscription) {
      this.authSubscription.unsubscribe();
    }
  }
}
