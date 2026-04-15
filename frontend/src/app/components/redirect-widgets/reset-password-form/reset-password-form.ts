import { ChangeDetectorRef, Component, OnInit } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { UsersService } from '../../../services/users/users-service';
import { TranslatePipe } from '@ngx-translate/core';
import {
  AbstractControl,
  FormBuilder,
  FormGroup,
  FormsModule,
  ReactiveFormsModule,
  ValidationErrors,
  ValidatorFn,
  Validators,
} from '@angular/forms';
import { ControlError } from '../../control-error/control-error';
import { UserResponseDto } from '../../../core/api/models/user-response-dto';
import { HttpErrorResponse } from '@angular/common/http';
import { FormSubmitDetail } from '../../form-submit-detail/form-submit-detail';

@Component({
  selector: 'app-forgot-password-form',
  imports: [TranslatePipe, FormsModule, ReactiveFormsModule, ControlError, FormSubmitDetail],
  templateUrl: './reset-password-form.html',
  styleUrl: './reset-password-form.css',
})
export class ResetPasswordForm implements OnInit {
  form: FormGroup;
  isPasswordVisible = false;
  isConfirmPasswordVisible = false;

  isSubmitting = false;
  success: boolean | null = null;
  submitStatusKey: string | null = null;

  constructor(
    private cdr: ChangeDetectorRef,
    private formBuilder: FormBuilder,
    private activatedRoute: ActivatedRoute,
    private router: Router,
    private usersService: UsersService,
  ) {
    this.form = formBuilder.group({
      password: ['', [Validators.required, Validators.minLength(6)]],
      confirmPassword: [
        '',
        [Validators.required, Validators.minLength(6), this.confirmPasswordValidator()],
      ],
    });
  }

  ngOnInit(): void {}

  public isControlRequired(controlName: string): boolean {
    const control = this.form.get(controlName);
    if (control && control.validator) {
      const validator = control.validator({} as AbstractControl);
      return validator && validator['required'];
    }
    return false;
  }

  public togglePasswordVisibility(): void {
    this.isPasswordVisible = !this.isPasswordVisible;
  }

  public toggleConfirmPasswordVisibility(): void {
    this.isConfirmPasswordVisible = !this.isConfirmPasswordVisible;
  }

  public isFormValid(): boolean {
    return this.form.valid && !this.isSubmitting;
  }

  public confirmPasswordValidator(): ValidatorFn {
    return (control: AbstractControl): ValidationErrors | null => {
      if (!control.parent) {
        return null;
      }
      const password = this.form.get('password');
      const confirmPassword = this.form.get('confirmPassword');
      if (!password || !confirmPassword) {
        return null;
      }
      return password.value !== confirmPassword.value ? { passwordMismatch: true } : null;
    };
  }

  public onSubmit(): void {
    const password: string = this.form.get('password')?.value!;
    const token: string | null = this.activatedRoute.snapshot.queryParamMap.get('token');
    if (!token) {
      this.router.navigate(['/portal/login']);
      return;
    }
    this.isSubmitting = true;
    this.form.markAsPristine();

    this.usersService.resetPassword(token, password).subscribe({
      next: (data: UserResponseDto): void => {
        this.success = true;
        this.submitStatusKey = 'submitSuccessfulMessage';
        this.cdr.detectChanges();

        setTimeout(() => {
          this.router.navigate(['/portal/login']);
        }, 5000);
      },
      error: (err: HttpErrorResponse): void => {
        this.success = false;
        if (err.status === 400 || err.status === 401) {
          this.submitStatusKey = 'resetPasswordInvalidToken';
        } else if (err.status === 404) {
          this.submitStatusKey = 'resetPasswordUnexistentUser';
        } else if (err.status === 500) {
          this.submitStatusKey = 'resetPasswordServerError';
        }
        this.isSubmitting = false;
        this.cdr.detectChanges();

        setTimeout(() => {
          this.router.navigate(['/portal/login']);
        }, 7000);
      },
    });
  }

  public getSubmitMessageKey(): string | null {
    if (!this.submitStatusKey) {
      return null;
    }

    const key = this.toKebabCase(this.submitStatusKey);
    const prefix = this.success === true ? 'reset-password-form' : 'error-messages';
    return `${prefix}.${key}`;
  }

  private toKebabCase(value: string): string {
    return value.replace(/([a-z])([A-Z])/g, '$1-$2').toLowerCase();
  }
}
