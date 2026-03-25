import { Component } from '@angular/core';
import { TranslatePipe } from '@ngx-translate/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-reset-password-invalid-token',
  imports: [TranslatePipe],
  templateUrl: './reset-password-invalid-token.html',
  styleUrl: './reset-password-invalid-token.css',
})
export class ResetPasswordInvalidToken {
  constructor(private router: Router) {}

  public returnToLoginClick(): void {
    this.router.navigate(['/portal/login']);
  }
}
