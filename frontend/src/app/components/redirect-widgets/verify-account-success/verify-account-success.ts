import { Component } from '@angular/core';
import { TranslatePipe } from '@ngx-translate/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-verify-account-success',
  imports: [TranslatePipe],
  templateUrl: './verify-account-success.html',
  styleUrl: './verify-account-success.css',
})
export class VerifyAccountSuccess {
  constructor(public router: Router) {}

  public returnToLoginClick(): void {
    this.router.navigate(['/portal/login']);
  }
}
