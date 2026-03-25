import { ChangeDetectorRef, Component, OnInit } from '@angular/core';
import { Router, RouterLink} from '@angular/router';
import { UsersService } from '../../../services/users/users-service';
import {ControlError} from "../../control-error/control-error";
import {FormSubmitDetail} from "../../form-submit-detail/form-submit-detail";
import { GoogleSigninButtonDirective } from '@abacritt/angularx-social-login';
import {
  AbstractControl,
  FormBuilder,
  FormGroup,
  ReactiveFormsModule,
  Validators,
} from '@angular/forms';
import {TranslatePipe} from "@ngx-translate/core";
import { HttpErrorResponse } from '@angular/common/http';
import { FormDescription } from '../form-description/form-description';

@Component({
  selector: 'app-forgot-password-form',
  imports: [
    ControlError,
    FormSubmitDetail,
    ReactiveFormsModule,
    RouterLink,
    TranslatePipe,
    FormDescription,
  ],
  standalone: true,
  templateUrl: './forgot-password-form.html',
  styleUrl: './forgot-password-form.css',
})
export class ForgotPasswordForm implements OnInit {
  form: FormGroup;
  success: boolean | null = null;
  submitMessage: string | null = null;
  isSubmitting: boolean = false;

  constructor(
    private formBuilder: FormBuilder,
    private cdr: ChangeDetectorRef,
    private usersService: UsersService,
  ) {
    this.form = this.formBuilder.group({
      email: ['', [Validators.required, Validators.email]],
    });
  }

  ngOnInit(): void {}

  public onSubmit(): void {
    const email: string | null = this.form.controls['email'].value;
    if (!email) {
      return;
    }
    this.isSubmitting = true;
    this.usersService.requestResetPassword(email!).subscribe({
      next: (data: any) => {
        this.success = true;
        this.submitMessage = 'forgot-password-form.forgot-password-submit-message-success';
        this.cdr.detectChanges();

        setTimeout(() => {
          this.isSubmitting = false;
          this.success = null;
          this.submitMessage = null;
          this.cdr.detectChanges();
        }, 5000);
      },
      error: (err: HttpErrorResponse) => {
        this.success = false;
        if (err.status === 403) {
          this.submitMessage = 'error-messages.forgot-password-unverified-user';
        } else if (err.status === 404) {
          this.submitMessage = 'error-messages.forgot-password-inexistent-user';
        } else if (err.status === 500) {
          this.submitMessage = 'error-messages.forgot-password-server-error';
        } else {
          this.submitMessage = 'error-messages.forgot-password-error';
        }
        this.cdr.detectChanges();
        setTimeout(() => {
          this.isSubmitting = false;
          this.success = null;
          this.submitMessage = null;
          this.cdr.detectChanges();
        }, 5000);
      },
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
    return this.form.valid;
  }
}
