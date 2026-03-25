import { Component } from '@angular/core';
import { TranslatePipe } from '@ngx-translate/core';
import { UsersService } from '../../services/users/users-service';
import { UserResponseDto } from '../../core/api/models/user-response-dto';

@Component({
  selector: 'app-topbar',
  imports: [TranslatePipe],
  templateUrl: './topbar.html',
  styleUrl: './topbar.css',
})
export class Topbar {
  email!: string;
  credits!: number;

  constructor(
    private usersService: UsersService,
  ) {

  }

  public logoutClick() {
    this.usersService.getTopbarInformation().subscribe({
      next: (data: UserResponseDto): void => {
        this.email = data.email;
        this.credits = data.credits!;
      }, error: err => {

      }
    })
  }
}
