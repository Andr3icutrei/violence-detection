import { Component, EventEmitter, OnInit, Output } from '@angular/core';
import {
  AbstractControl,
  FormBuilder,
  FormGroup,
  ReactiveFormsModule, ValidationErrors,
  ValidatorFn,
  Validators,
} from '@angular/forms';
import { ControlError } from '../../control-error/control-error';
import { TranslatePipe } from '@ngx-translate/core';
import { PortalPage } from '../portal-page';
import { AuthService } from '../../../services/auth/auth-service';
import { UsersService } from '../../../services/users/users-service';
import { UserResponseDto } from '../../../core/api/models/user-response-dto';
import { Router } from '@angular/router';
import { HttpErrorResponse } from '@angular/common/http';

@Component({
  selector: 'app-register-form',
  imports: [ReactiveFormsModule, ControlError, TranslatePipe],
  templateUrl: './register-form.html',
  styleUrl: './register-form.css',
})
export class RegisterForm implements OnInit {
  form: FormGroup;
  isPasswordVisible: boolean = false;
  isConfirmPasswordVisible: boolean = false;
  isSubmitting: boolean = false;

  @Output() switch = new EventEmitter<PortalPage>();

  constructor(
    private formBuilder: FormBuilder,
    private usersService: UsersService,
    private router: Router
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
      if(
        this.form.hasError('serverError') ||
        this.form.hasError('accountAlreadyExists')
      ) {
        this.form.setErrors(null);
        this.form.updateValueAndValidity({ emitEvent: false });
      }
    })
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

      return password !== confirmPassword ? { passwordMismatch: true } : null;
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

  public switchForm(form: PortalPage): void {
    this.switch.emit(form);
  }

  public submit(): void {
    const email:string = this.form.get('email')?.value!;
    const password:string = this.form.get('password')?.value!;

    this.isSubmitting = true;

    this.usersService.register(email, password).subscribe({
      next: (data: UserResponseDto): void => {
        this.switch.emit('login');
      },
      error: (err: HttpErrorResponse): void => {
        if(err.status === 409) {
          this.form.setErrors({ accountAlreadyExists: true });
        } else if (err.status === 500) {
          this.form.setErrors({ serverError: true });
        }
      },
      complete: (): void => {
        this.isSubmitting = false;
      }
    })
  }
}
