import { Component, OnInit } from '@angular/core';
import { UsersService } from '../../services/users/users-service';
import { ActivatedRoute, Router, RouterOutlet } from '@angular/router';
import { UserResponseDto } from '../../core/api/models/user-response-dto';
import { HttpErrorResponse } from '@angular/common/http';

@Component({
  selector: 'app-verify-account-page',
  imports: [RouterOutlet],
  standalone: true,
  templateUrl: './verify-account-page.html',
  styleUrl: './verify-account-page.css',
})
export class VerifyAccountPage implements OnInit {
  constructor(
    private activatedRoute: ActivatedRoute,
    private router: Router,
    private usersService: UsersService,
  ) {}

  ngOnInit(): void {
    const token: string | null = this.activatedRoute.snapshot.queryParamMap.get('token');
    if (!token) {
      this.router.navigate(['/portal/login']);
      return;
    }

    this.usersService.verifyAccount(token!).subscribe({
      next: (_data: UserResponseDto) => {
        this.router.navigate(['/verify-account/success']);
      },
      error: (err: HttpErrorResponse) => {
        if (err.status === 401 || err.status === 400) {
          this.router.navigate(['/verify-account/invalid-token']);
        } else if (err.status === 404) {
          this.router.navigate(['/verify-account/unexistent-user']);
        } else if (err.status === 409) {
          this.router.navigate(['/verify-account/already-verified']);
        } else if (err.status === 500) {
          this.router.navigate(['/verify-account/server-error']);
        }
      },
    });
  }
}
