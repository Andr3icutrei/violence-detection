import { Component } from '@angular/core';
import { TranslatePipe } from '@ngx-translate/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-verify-account-server-error',
  imports: [TranslatePipe],
  templateUrl: './verify-account-server-error.html',
  styleUrl: './verify-account-server-error.css',
})
export class VerifyAccountServerError {
  constructor(public router: Router) {
  }

  public returnToLoginClick(): void {
    this.router.navigate(['/portal/login']);
  }
}
