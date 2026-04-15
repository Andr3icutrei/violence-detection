import { ChangeDetectorRef, Component, EventEmitter, OnInit, Output } from '@angular/core';
import {
  AbstractControl,
  FormBuilder,
  FormGroup,
  ReactiveFormsModule,
  ValidationErrors,
  ValidatorFn,
  Validators,
} from '@angular/forms';
import { ControlError } from '../../control-error/control-error';
import { TranslatePipe } from '@ngx-translate/core';
import { PortalForm } from '../portal-form.type';
import { AuthService } from '../../../services/auth/auth-service';
import { UsersService } from '../../../services/users/users-service';
import { UserResponseDto } from '../../../core/api/models/user-response-dto';
import { Router, RouterLink } from '@angular/router';
import { HttpErrorResponse } from '@angular/common/http';
import { FormSubmitDetail } from '../../form-submit-detail/form-submit-detail';
import { FormDescription } from '../form-description/form-description';

@Component({
  selector: 'app-register-form',
  imports: [
    ReactiveFormsModule,
    TranslatePipe,
    ControlError,
    FormSubmitDetail,
    RouterLink,
    FormDescription,
  ],
  templateUrl: './register-form.html',
  styleUrl: './register-form.css',
})
export class RegisterForm implements OnInit {
  form: FormGroup;
  isPasswordVisible: boolean = false;
  isConfirmPasswordVisible: boolean = false;

  isSubmitting: boolean = false;
  submitStatusKey: string | null = null;
  success: boolean | null = null;

  @Output() switch: EventEmitter<PortalForm> = new EventEmitter<PortalForm>();

  constructor(
    private formBuilder: FormBuilder,
    private router: Router,
    private cdr: ChangeDetectorRef,
    private usersService: UsersService,
  ) {
    this.form = this.formBuilder.group({
      email: ['', [Validators.required, Validators.email]],
      password: ['', [Validators.required, Validators.minLength(6)]],
      confirmPassword: [
        '',
        [Validators.required, Validators.minLength(6), this.confirmPasswordValidator()],
      ],
    });
  }

  public ngOnInit(): void {
    this.form.valueChanges.subscribe(() => {
      if (
        this.form.dirty &&
        (this.form.hasError('serverError') || this.form.hasError('accountAlreadyExists'))
      ) {
        this.form.setErrors(null);
        this.form.updateValueAndValidity({ emitEvent: false });
      }
    });
  }

  private confirmPasswordValidator(): ValidatorFn {
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

  public isControlRequired(controlName: string) {
    if (this.form.contains(controlName)) {
      const control = this.form.get(controlName);
      return control?.hasValidator(Validators.required) || false;
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

  public switchForm(form: PortalForm): void {
    this.switch.emit(form);
  }

  public submit(): void {
    const email: string = this.form.get('email')?.value!;
    const password: string = this.form.get('password')?.value!;
    this.isSubmitting = true;
    this.form.markAsPristine();

    this.usersService.register(email, password).subscribe({
      next: (data: UserResponseDto): void => {
        this.success = true;
        this.submitStatusKey = 'submitSuccessfulMessage';
        this.cdr.detectChanges();

        setTimeout(() => {
          this.submitStatusKey = null;
          this.isSubmitting = false;
          this.cdr.detectChanges();
        }, 5000);
      },
      error: (err: HttpErrorResponse): void => {
        this.success = false;

        if (err.status === 409) {
          this.form.setErrors({ accountAlreadyExists: true });
          this.submitStatusKey = 'accountAlreadyExists';
        } else if (err.status === 500) {
          this.form.setErrors({ serverError: true });
          this.submitStatusKey = 'serverError';
        }
        this.cdr.detectChanges();

        setTimeout(() => {
          this.submitStatusKey = null;
          this.isSubmitting = false;
          this.form.setErrors(null);
          this.cdr.detectChanges();
        }, 5000);
      },
    });
  }

  public getSubmitMessageKey(): string | null {
    if (!this.submitStatusKey) {
      return null;
    }

    const key = this.toKebabCase(this.submitStatusKey);
    const prefix = this.success === true ? 'register-form' : 'error-messages';
    return `${prefix}.${key}`;
  }

  private toKebabCase(value: string): string {
    return value.replace(/([a-z])([A-Z])/g, '$1-$2').toLowerCase();
  }
}
