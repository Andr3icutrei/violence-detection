import { Component, input, Input } from '@angular/core';
import { VerifyAccountStatus } from '../verify-account-status.type';
import { ActivatedRoute, Router } from '@angular/router';
import { TranslatePipe } from '@ngx-translate/core';
import { UsersService } from '../../../services/users/users-service';
import { UserResponseDto } from '../../../core/api/models/user-response-dto';
import { HttpErrorResponse } from '@angular/common/http';

@Component({
  selector: 'app-verify-account-details',
  imports: [TranslatePipe],
  templateUrl: './verify-account-details.html',
  styleUrl: './verify-account-details.css',
})
export class VerifyAccountDetails {
  @Input() status: VerifyAccountStatus = 'success';
  @Input() icon!: string;
  @Input() showReturnToLogin: boolean = false;
  resentVerificationEmailMessage: string | null = null;

  constructor(
    private router: Router,
    private usersService: UsersService,
    private activatedRoute: ActivatedRoute,
  ) {

  }

  public returnToLoginClick(): void {
    if (!this.showReturnToLogin) {
      return;
    }

    this.router.navigate(['login']);
  }

  public getSuccessfulMessage(): string {
    return this.status === 'success' ? 'success-message' : 'error-message';
  }

  public resendVerificationEmail(): void {
    const token: string | null = this.activatedRoute.snapshot.queryParamMap.get('token');
    if (!token) {
      this.router.navigate(['login']);
      return;
    }

    this.usersService.resendVerificationEmail(token).subscribe({
      next: (data: UserResponseDto): void => {
        this.resentVerificationEmailMessage = 'verify-account.resent-verification-email';
      },
      error: (err: HttpErrorResponse): void => {
        if (err.status === 400) {
          this.resentVerificationEmailMessage = 'verify-account.invalid-token-resent';
        } else if (err.status === 404) {
          this.resentVerificationEmailMessage = 'verify-account.unexisting-user-resent';
        } else if (err.status === 409) {
          this.resentVerificationEmailMessage = 'verify-account.already-verified-resent';
        } else if (err.status === 500) {
          this.resentVerificationEmailMessage = 'verify-account.server-error-resent';
        }
      }
    })
  }
}
