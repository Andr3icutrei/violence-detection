import { Component } from '@angular/core';
import { TranslatePipe } from '@ngx-translate/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-verify-account-already-verified',
  imports: [TranslatePipe],
  templateUrl: './verify-account-already-verified.html',
  styleUrl: './verify-account-already-verified.css',
})
export class VerifyAccountAlreadyVerified {
  constructor(public router: Router) {

  }

  public returnToLoginClick(): void {
    this.router.navigate(['/portal/login']);
  }
}
