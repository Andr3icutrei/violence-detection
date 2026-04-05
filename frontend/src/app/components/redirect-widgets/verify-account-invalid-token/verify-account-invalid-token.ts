import { Component } from '@angular/core';
import { TranslatePipe } from '@ngx-translate/core';
import { UsersService } from '../../../services/users/users-service';
import { ActivatedRoute, Router } from '@angular/router';
import { HttpErrorResponse } from '@angular/common/http';

@Component({
  selector: 'app-verify-account-invalid-token',
  imports: [TranslatePipe],
  templateUrl: './verify-account-invalid-token.html',
  styleUrl: './verify-account-invalid-token.css',
})
export class VerifyAccountInvalidToken {
  resentVerificationEmailMessage: string | null = null;
  success: boolean | null = null;

  constructor(
    public router: Router,
    private usersService: UsersService,
    private activatedRoute: ActivatedRoute,
  ) {}

  public returnToLoginClick(): void {
    this.router.navigate(['/portal/login']);
  }

  public resendVerificationEmail(): void {
    const token: string | null = this.activatedRoute.snapshot.queryParamMap.get('token');
    console.log(token);
    if (!token) {
      this.router.navigate(['/portal/login']);
      return;
    }
    this.usersService.resendVerificationEmail(token).subscribe({
      next: (): void => {
        this.resentVerificationEmailMessage = 'verify-account.resent-verification-email';
        this.success = true;
      },
      error: (err: HttpErrorResponse): void => {
        this.resentVerificationEmailMessage = 'verify-account.resend-verification-email-failed';
        this.success = false;
      }
    });
  }
}
