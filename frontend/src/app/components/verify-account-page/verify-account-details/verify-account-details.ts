import { Component, Input } from '@angular/core';
import { VerifyAccountStatus } from '../verify-account-status.type';
import { Router } from '@angular/router';
import { TranslatePipe } from '@ngx-translate/core';

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

  constructor(private router: Router) {}

  public returnToLoginClick(): void {
    if (!this.showReturnToLogin) {
      return;
    }

    this.router.navigate(['login']);
  }

  getSuccessfulMessage(): string {
    return this.status === 'success' ? 'success-message' : 'error-message';
  }
}
