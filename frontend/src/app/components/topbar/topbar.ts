import { ChangeDetectorRef, Component, OnInit } from '@angular/core';
import { TranslatePipe } from '@ngx-translate/core';
import { UsersService } from '../../services/users/users-service';
import { UserResponseDto } from '../../core/api/models/user-response-dto';

@Component({
  selector: 'app-topbar',
  imports: [TranslatePipe],
  templateUrl: './topbar.html',
  styleUrl: './topbar.css',
})
export class Topbar implements OnInit {
  email!: string;
  credits!: number;
  emailInitials!: string;
  isDropdownOpen: boolean = false;

  constructor(
    private cdr: ChangeDetectorRef,
    private usersService: UsersService
  ) {

  }

  ngOnInit(): void {
    this.usersService.getTopbarInformation().subscribe({
      next: (data: UserResponseDto): void => {
        this.email = data.email;
        this.credits = data.credits!;
        this.emailInitials = this.email.slice(0, 2).toUpperCase();
        this.cdr.detectChanges();
      },
      error: (err): void => {
        console.error('Failed to load topbar information.', err);
      },
    });
  }

  public logoutClick() {}

  public toggleDropdown(): void {
    this.isDropdownOpen = !this.isDropdownOpen;
  }
}
