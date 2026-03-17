import { Component, OnInit } from '@angular/core';
import { UsersService } from '../../services/users/users-service';
import { ActivatedRoute, Router } from '@angular/router';
import { UserResponseDto } from '../../core/api/models/user-response-dto';
import { HttpErrorResponse } from '@angular/common/http';
import { VerifyAccountDetails } from './verify-account-details/verify-account-details';
import { VerifyAccountStatus } from './verify-account-status.type';

@Component({
  selector: 'app-verify-account-page',
  imports: [
    VerifyAccountDetails
  ],
  standalone: true,
  templateUrl: './verify-account-page.html',
  styleUrl: './verify-account-page.css',
})
export class VerifyAccountPage implements OnInit {
  status: VerifyAccountStatus | null = null;
  icon: string | null = null;

  constructor(
    private activatedRoute: ActivatedRoute,
    private router: Router,
    private usersService: UsersService,
  ) {}

  ngOnInit(): void {
    const token: string | null = this.activatedRoute.snapshot.params['token'];
    if (token === null) {
      this.router.navigate(['login']);
      return;
    }

    this.usersService.verifyAccount(token!).subscribe({
      next: (data: UserResponseDto) => {
        this.status = 'success';
        this.icon = 'fa-light fa-party-horn';
      },
      error: (err: HttpErrorResponse) => {
        if (err.status === 400) {
          this.status = 'invalid-token';
        } else if (err.status === 404) {
          this.status = 'unexisting-user';
        } else if (err.status === 409) {
          this.status = 'already-verified';
        } else if (err.status === 500) {
          this.status = 'server-error';
        }
        this.icon = 'fa-solid fa-bug';
      },
    });
  }
}
