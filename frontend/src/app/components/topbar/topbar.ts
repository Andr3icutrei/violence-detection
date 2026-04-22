import { ChangeDetectorRef, Component, OnDestroy, OnInit } from '@angular/core';
import { TranslatePipe } from '@ngx-translate/core';
import { UsersService } from '../../services/users/users-service';
import { UserResponseDto } from '../../core/api/models/user-response-dto';
import { TopbarRefreshService } from '../../services/users/topbar-refresh.service';
import { Subscription } from 'rxjs';
import { AuthService } from '../../services/auth/auth-service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-topbar',
  imports: [TranslatePipe],
  templateUrl: './topbar.html',
  styleUrl: './topbar.css',
})
export class Topbar implements OnInit, OnDestroy {
  email!: string;
  credits!: number;
  emailInitials!: string;
  isAdmin!: boolean;
  isDropdownOpen: boolean = false;
  private refreshSubscription?: Subscription;

  constructor(
    private cdr: ChangeDetectorRef,
    private usersService: UsersService,
    private topbarRefreshService: TopbarRefreshService,
    private authService: AuthService,
    private router: Router,
  ) {

  }

  ngOnInit(): void {
    this.loadTopbarInformation();

    this.refreshSubscription = this.topbarRefreshService.refresh$.subscribe(() => {
      this.loadTopbarInformation();
    });
  }

  ngOnDestroy(): void {
    this.refreshSubscription?.unsubscribe();
  }

  private loadTopbarInformation(): void {
    this.usersService.getTopbarInformation().subscribe({
      next: (data: UserResponseDto): void => {
        this.email = data.email;
        this.credits = data.credits!;
        this.emailInitials = this.email.slice(0, 2).toUpperCase();
        this.isAdmin = data.is_admin!;
        this.cdr.detectChanges();
      },
      error: (err): void => {
        console.error('Failed to load topbar information.', err);
      },
    });
  }

  public logoutClick(): void {
    this.authService.logout().subscribe({
      next: () => {
        this.router.navigate(['/']);
      },
      error: () => {
        this.router.navigate(['/']);
      }
    })
  }

  public toggleDropdown(): void {
    this.isDropdownOpen = !this.isDropdownOpen;
  }
}
