import { Component, OnInit } from '@angular/core';
import { ActivatedRoute, Router, RouterOutlet } from '@angular/router';
import { UsersService } from '../../services/users/users-service';

@Component({
  selector: 'app-reset-password-page',
  imports: [RouterOutlet],
  templateUrl: './reset-password-page.html',
  styleUrl: './reset-password-page.css',
})
export class ResetPasswordPage implements OnInit {
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

    this.usersService.verifyResetPasswordToken(token!).subscribe({
      next: (): void => {
        this.router.navigate(['/reset-password/reset'], { queryParams: { token } });
      },
      error: (): void => {
        this.router.navigate(['/reset-password/invalid-token']);
      },
    });
  }
}
