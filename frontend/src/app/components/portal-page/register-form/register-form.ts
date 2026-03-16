import { Component, EventEmitter, Output } from '@angular/core';
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

@Component({
  selector: 'app-register-form',
  imports: [ReactiveFormsModule, ControlError, TranslatePipe],
  templateUrl: './register-form.html',
  styleUrl: './register-form.css',
})
export class RegisterForm {
  form: FormGroup;
  isPasswordVisible: boolean = false;
  isConfirmPasswordVisible: boolean = false;
  isSubmitted: boolean = false;

  @Output() switch = new EventEmitter<PortalPage>();

  constructor(private formBuilder: FormBuilder) {
    this.form = this.formBuilder.group({
      email: ['', [Validators.required, Validators.email]],
      password: ['', [Validators.required, Validators.minLength(6)]],
      confirmPassword: [
        '',
        [Validators.required, Validators.minLength(6), this.confirmPasswordValidator()],
      ],
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

  public togglePasswordVisibilty(): void {
    this.isPasswordVisible = !this.isPasswordVisible;
  }

  public toggleConfirmPasswordVisibility(): void {
    this.isConfirmPasswordVisible = !this.isConfirmPasswordVisible;
  }

  public isFormValid(): boolean {
    return this.form.valid && !this.isSubmitted;
  }

  public switchForm(form: PortalPage): void {
    this.switch.emit(form);
  }

  public submit(): void {}
}
