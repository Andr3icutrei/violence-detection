import { Component } from '@angular/core';
import { TranslatePipe } from '@ngx-translate/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-verify-account-unexistent-user',
  imports: [TranslatePipe],
  templateUrl: './verify-account-unexistent-user.html',
  styleUrl: './verify-account-unexistent-user.css',
})
export class VerifyAccountUnexistentUser {
  constructor(public router: Router) {
  }

  public returnToLoginClick(): void {
    this.router.navigate(['/portal/login']);
  }
}
